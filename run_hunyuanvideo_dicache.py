import os
import time
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens
from typing import Any, List, Tuple, Optional, Union, Dict
import torch
import json
import numpy as np
import torch.nn.functional as F
import argparse
import pdb
import matplotlib.pyplot as plt


def _slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().replace("/", "_")
    s = "".join(ch if (ch.isalnum() or ch in (" ", "_", "-", ".")) else "_" for ch in s)
    s = "_".join(s.split())
    return s[:max_len] if len(s) > max_len else s


def _parse_dicache_cli(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse DiCache-only flags from argv, leaving the rest for hyvideo.config.parse_args().
    We do this here (instead of editing hyvideo/config.py) to keep upstream args untouched.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dicache-trace", action="store_true", help="Enable per-timestep probe tracing + plot/json outputs.")
    p.add_argument(
        "--dicache-trace-only",
        action="store_true",
        help="Trace-only calibration: record accumulated probe error but do not make any skip/reset decisions.",
    )
    p.add_argument(
        "--dicache-ret-ratio",
        type=float,
        default=0.2,
        help="Retention prefix ratio (forced full-run warmup) used by DiCache skip logic. Ignored when --dicache-trace-only is set.",
    )
    p.add_argument("--dicache-probe-depth", type=int, default=1, help="Number of early double_blocks to run as probe (recommend 1~5).")
    p.add_argument("--dicache-rel-l1-thresh", type=float, default=0.1, help="Fixed threshold for DiCache skip logic (accumulated probe error).")
    p.add_argument(
        "--dicache-results-root",
        type=str,
        default="./dicache_results",
        help="Root directory for all DiCache artifacts (videos/, logs/, calibration/).",
    )
    return p.parse_known_args(argv)


def _resolve_dicache_dirs(dicache_results_root: str, save_path: str) -> Tuple[Path, Path, Path]:
    """
    Unify DiCache outputs under one root:
      <root>/videos/        (sample mp4s)
      <root>/calibration/   (D1 json/png curves)
      <root>/logs/          (tmux/stdout logs; external but we provide the path)
    If user passes --save-path explicitly, we still write calibration under <root>/calibration.
    """
    root = Path(dicache_results_root).expanduser().resolve()
    videos_dir = root / "videos"
    calibration_dir = root / "calibration"
    logs_dir = root / "logs"
    return root, videos_dir, calibration_dir


def dicache_forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,  # Should be in range(0, 1000).
    text_states: torch.Tensor = None,
    text_mask: torch.Tensor = None,  # Now we don't use it.
    text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
    freqs_cos: Optional[torch.Tensor] = None,
    freqs_sin: Optional[torch.Tensor] = None,
    guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
    return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        skip_forward = False
        
        # --------------------- Online Probe Profiling Scheme -----------------------
        if (getattr(self, "trace_enabled", False) or self.cnt >= int(self.ret_ratio * self.num_steps)):
            test_img, test_txt = img.clone(), txt.clone()
            probe_blocks = self.double_blocks[0:self.probe_depth]
            for probe_block in probe_blocks:
                test_double_block_args = [
                    test_img,
                    test_txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                ]
                test_img, test_txt = probe_block(*test_double_block_args)

            # Safe init at first step (or if state got cleared)
            if getattr(self, "previous_input", None) is None:
                self.previous_input = img.detach()
            if getattr(self, "previous_probe_img", None) is None:
                self.previous_probe_img = test_img.detach()

            prev_in = self.previous_input
            prev_probe = self.previous_probe_img
            eps = 1e-12
            delta_x = (img - prev_in).abs().mean() / (prev_in.abs().mean() + eps)
            delta_y = (test_img - prev_probe).abs().mean() / (prev_probe.abs().mean() + eps)

            # Trace-only calibration: record a running sum without any resets/skip decisions.
            if getattr(self, "dicache_trace_only", False):
                cls = self.__class__
                if not hasattr(cls, "trace_steps"):
                    cls.trace_steps = []
                    cls.trace_delta_y = []
                    cls.trace_accumulated = []
                    cls.trace_delta_x = []
                    cls.trace_prompt = getattr(self, "trace_prompt", "")
                if not hasattr(cls, "trace_running_sum"):
                    cls.trace_running_sum = 0.0
                cls.trace_running_sum += float(delta_y.detach().cpu().item())
                cls.trace_steps.append(int(self.cnt))
                cls.trace_delta_y.append(float(delta_y.detach().cpu().item()))
                cls.trace_accumulated.append(float(cls.trace_running_sum))
                cls.trace_delta_x.append(float(delta_x.detach().cpu().item()))
            else:
                # Original skip / resume logic.
                # Guard: never skip if residual_cache is None (no full run yet).
                # This is the correct minimal check instead of a coarse ret_ratio warmup.
                self.accumulated_rel_l1_distance += delta_y
                _thresh = (
                    self.rel_l1_thresh[self.cnt]
                    if isinstance(self.rel_l1_thresh, list)
                    else self.rel_l1_thresh
                )
                if (self.accumulated_rel_l1_distance <= _thresh
                        and self.residual_cache is not None):
                    skip_forward = True
                    self.resume_flag = False
                else:  # continue calculating, but can resume from the probe state
                    self.accumulated_rel_l1_distance = 0
                    self.resume_flag = True
            
        # ---------------------------------------------------------------------------

        if skip_forward: 
            ori_img = img.clone()

            # --------------------- Dynamic Cache Trajectory Alignment --------------
            if len(self.residual_window) >= 2:
                current_residual_indicator = test_img - img
                gamma = ((current_residual_indicator - self.probe_residual_window[-2]).abs().mean() / (self.probe_residual_window[-1] - self.probe_residual_window[-2]).abs().mean()).clip(1, 1.5)
                img = img + self.residual_window[-2] + gamma * (self.residual_window[-1] - self.residual_window[-2])
            else:
                img = img + self.residual_cache
            # -----------------------------------------------------------------------

            self.previous_probe_img = test_img
            self.previous_input = ori_img
        else:
            # -------------- resume from previously calculated result --------------
            ori_img = img
            if self.resume_flag:
                img = test_img
                txt = test_txt
                unpass_blocks = self.double_blocks[self.probe_depth:]
            else:
                unpass_blocks = self.double_blocks
            # ----------------------------------------------------------------------

            # --------------------- Pass through DiT blocks ------------------------
            for index_block, block in enumerate(unpass_blocks): # self.double_blocks
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                ]

                img, txt = block(*double_block_args)

                # ----------------- record probe feature ---------------
                if index_block == self.probe_depth - 1:
                    if self.cnt >= int(self.ret_ratio*self.num_steps):
                        self.previous_probe_img = test_img
                    else:
                        self.previous_probe_img = img
                # ---------------------------------------------------------------

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                    ]

                    x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]
            self.residual_cache = img - ori_img # residual between block 0 and block M 
            self.probe_residual_cache = self.previous_probe_img - ori_img # residual between block 0 and block m 
            self.previous_input = ori_img

            if len(self.residual_window) <= 2:
                self.residual_window.append(self.residual_cache)
                self.probe_residual_window.append(self.probe_residual_cache)
            else:
                self.residual_window[-2] = self.residual_window[-1]
                self.residual_window[-1] = self.residual_cache
                self.probe_residual_window[-2] = self.probe_residual_window[-1]
                self.probe_residual_window[-1] = self.probe_residual_cache
       
        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        img = self.unpatchify(img, tt, th, tw)
        
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
            self.accumulated_rel_l1_distance = 0
            self.resume_flag = False  
            self.residual_window = [] 
            self.probe_residual_window = []
            cls = self.__class__
            if hasattr(cls, "trace_running_sum"):
                cls.trace_running_sum = 0.0
        
        if return_dict:
            out["x"] = img
            return out
        return img

def main():
    dicache_args, remaining = _parse_dicache_cli(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    dicache_root, dicache_videos_dir, dicache_calibration_dir = _resolve_dicache_dirs(
        dicache_args.dicache_results_root, args.save_path
    )

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args
    
    hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.probe_depth = int(dicache_args.dicache_probe_depth)  # recommend 1~5
    hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
    hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh = float(dicache_args.dicache_rel_l1_thresh)
    hunyuan_video_sampler.pipeline.transformer.__class__.ret_ratio = float(dicache_args.dicache_ret_ratio)
    hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.residual_cache = None
    hunyuan_video_sampler.pipeline.transformer.__class__.probe_residual_cache = None
    hunyuan_video_sampler.pipeline.transformer.__class__.residual_window = []
    hunyuan_video_sampler.pipeline.transformer.__class__.probe_residual_window = []
    hunyuan_video_sampler.pipeline.transformer.__class__.resume_flag = False
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_input = None
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_probe_img = None

    # D1 calibration trace (baseline full-run): log probe L1_rel curve + accumulated sum
    trace_enabled = bool(dicache_args.dicache_trace)
    hunyuan_video_sampler.pipeline.transformer.__class__.trace_enabled = trace_enabled
    hunyuan_video_sampler.pipeline.transformer.__class__.dicache_trace_only = bool(dicache_args.dicache_trace_only)
    hunyuan_video_sampler.pipeline.transformer.__class__.trace_prompt = args.prompt
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = dicache_forward 

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )

    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) * 1e-3

    samples = outputs['samples']
    
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_path}/DiCache_{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=16)
            logger.info(f'Sample save to: {save_path}')

        # Write D1 calibration artifacts after generation
        if trace_enabled and hasattr(hunyuan_video_sampler.pipeline.transformer.__class__, "trace_steps"):
            dicache_calibration_dir.mkdir(parents=True, exist_ok=True)
            prompt_slug = _slugify(args.prompt, max_len=60)
            stamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
            json_path = dicache_calibration_dir / f"dicache_probe_curve_hunyuanvideo_544p_{stamp}_{prompt_slug}.json"
            png_path = dicache_calibration_dir / f"dicache_probe_curve_hunyuanvideo_544p_{stamp}_{prompt_slug}.png"

            steps = hunyuan_video_sampler.pipeline.transformer.__class__.trace_steps
            delta_y = hunyuan_video_sampler.pipeline.transformer.__class__.trace_delta_y
            accum = hunyuan_video_sampler.pipeline.transformer.__class__.trace_accumulated
            delta_x = hunyuan_video_sampler.pipeline.transformer.__class__.trace_delta_x

            payload = {
                "prompt": args.prompt,
                "seed": int(args.seed),
                "infer_steps": int(args.infer_steps),
                "ret_ratio": float(hunyuan_video_sampler.pipeline.transformer.__class__.ret_ratio),
                "probe_depth": int(hunyuan_video_sampler.pipeline.transformer.__class__.probe_depth),
                "accumulated_definition": "E(t)=sum_{i<=t} delta_y(i)",
                "delta_y_definition": "delta_y(t)=L1_rel(probe_out_t, probe_out_{t-1})",
                "delta_x_definition": "delta_x(t)=L1_rel(model_input_t, model_input_{t-1})",
                "steps": steps,
                "delta_y": delta_y,
                "accumulated": accum,
                "delta_x": delta_x,
            }
            json_path.write_text(json.dumps(payload, indent=2))

            plt.figure(figsize=(10, 4.5))
            plt.plot(steps, delta_y, linewidth=2.0)
            plt.xlabel("Timestep (denoising step index)")
            plt.ylabel("Probe change being accumulated: L1_rel(probe_t, probe_{t-1})")
            plt.title("DiCache D1 calibration: per-step probe-change (baseline full-run)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=200)
            plt.close()

            logger.info(f"DiCache trace JSON saved to: {str(json_path)}")
            logger.info(f"DiCache trace plot saved to: {str(png_path)}")


if __name__ == "__main__":
    main()
