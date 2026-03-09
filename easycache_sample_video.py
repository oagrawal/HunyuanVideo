# Copyright 2025 The Tecent Hunyuan Team Authors. All rights reserved.
# Copyright 2025 The Huazhong University of Science and Technology VLRLab Authors. All rights reserved.

import os
import time
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
import portalocker
import matplotlib.pyplot as plt


def easycache_baseline_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Baseline (no-caching) forward pass that records EasyCache's metric k_t
    at every step. k_t = ||v_t - v_{t-1}|| / ||x_t - x_{t-1}|| measures how
    much the model amplifies input changes — the signal EasyCache accumulates
    to decide when to skip. pred_change_t = k_{t-1} * (||x_t - x_{t-1}|| /
    ||v_{t-1}||) is the per-step contribution to the accumulated error.
    """
    torch.cuda.synchronize()
    start_time = time.time()

    out = {}
    raw_input = x.clone()
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
    vec = vec + self.vector_in(text_states_2)

    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(guidance)

    # ------------------------------------------------------------------
    # Compute pred_change using k from the previous step.
    # This is what EasyCache accumulates toward its threshold, so plotting
    # it shows how much "work" each step would contribute to caching decisions.
    # ------------------------------------------------------------------
    if self.previous_raw_input is not None and self.previous_output is not None \
            and self.k is not None:
        raw_input_change = (raw_input - self.previous_raw_input).abs().mean()
        output_norm = self.previous_output.abs().mean()
        pred_change = (self.k * (raw_input_change / output_norm)).item()
        self.pred_change_history.append(pred_change)

    self.previous_raw_input = raw_input.clone()

    # ------------------------------------------------------------------
    # Full model compute (no skipping)
    # ------------------------------------------------------------------
    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q
    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    for _, block in enumerate(self.double_blocks):
        img, txt = block(img, txt, vec, cu_seqlens_q, cu_seqlens_kv,
                         max_seqlen_q, max_seqlen_kv, freqs_cis)

    x = torch.cat((img, txt), 1)
    if len(self.single_blocks) > 0:
        for _, block in enumerate(self.single_blocks):
            x = block(x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv,
                      max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin))

    img = x[:, :img_seq_len, ...]
    img = self.final_layer(img, vec)
    result = self.unpatchify(img, tt, th, tw)

    # ------------------------------------------------------------------
    # Update k_t = ||v_t - v_{t-1}|| / ||x_t - x_{t-1}||
    # This is EasyCache's core "transformation rate" metric.
    # ------------------------------------------------------------------
    if self.previous_output is not None and self.prev_prev_raw_input is not None:
        output_change = (result - self.previous_output).abs().mean()
        input_change = (self.previous_raw_input - self.prev_prev_raw_input).abs().mean()
        if input_change > 0:
            self.k = output_change / input_change
            self.k_history.append(self.k.item())

    self.prev_prev_raw_input = self.previous_raw_input
    self.previous_output = result.clone()

    self.cnt += 1

    torch.cuda.synchronize()
    end_time = time.time()
    self.total_time += (end_time - start_time)

    if return_dict:
        out["x"] = result
        return out
    return result


def main():
    args = parse_args()

    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    os.makedirs(args.save_path, exist_ok=True)

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args

    transformer = hunyuan_video_sampler.pipeline.transformer
    transformer.__class__.forward = easycache_baseline_forward

    # State for k_t computation
    transformer.__class__.cnt = 0
    transformer.__class__.total_time = 0.0
    transformer.__class__.k = None
    transformer.__class__.previous_raw_input = None
    transformer.__class__.previous_output = None
    transformer.__class__.prev_prev_raw_input = None

    # Profiling lists
    transformer.__class__.k_history = []
    transformer.__class__.pred_change_history = []

    transformer.total_time = 0.0

    start_time = time.time()
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=1,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    end_time = time.time()
    e2e_time = end_time - start_time
    logger.info(f"End-to-end generation time: {e2e_time:.2f} seconds")

    samples = outputs['samples']

    # ---- Build output folder ----
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    seed = outputs['seeds'][0]
    prompt_short = outputs['prompts'][0][:50].replace('/', '').replace(' ', '_')
    generation_folder = f"baseline_profile_{time_flag}_seed{seed}_{prompt_short}"
    save_path = os.path.join(args.save_path, generation_folder)
    os.makedirs(save_path, exist_ok=True)

    # ---- Save video ----
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            video_path = os.path.join(save_path, 'video.mp4')
            save_videos_grid(samples[i].unsqueeze(0), video_path, fps=24)
            logger.info(f'Sample saved to: {video_path}')

    k_history = transformer.k_history
    pred_change_history = transformer.pred_change_history
    steps = args.infer_steps

    # ---- Plot k_t ----
    if len(k_history) > 0:
        # k_history starts from step 3 (needs two previous inputs + one previous output)
        # offset so the x-axis reflects actual step numbers
        k_start_step = steps - len(k_history)
        x_k = range(k_start_step, steps)

        plt.figure(figsize=(10, 5))
        plt.plot(x_k, k_history, 'b-', linewidth=2, marker='o', markersize=5)
        plt.xlabel('Diffusion Step')
        plt.ylabel('k_t')
        plt.title('EasyCache Transformation Rate  k_t = ||v_t − v_{t-1}|| / ||x_t − x_{t-1}||\n'
                  '(baseline run — no caching)')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(max(1, steps // 10)))
        plot_path = os.path.join(save_path, 'k_t_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'k_t plot saved to: {plot_path}')

        with open(os.path.join(save_path, 'k_t.txt'), 'w') as f:
            for v in k_history:
                f.write(f"{v}\n")

    # ---- Plot pred_change ----
    if len(pred_change_history) > 0:
        pc_start_step = steps - len(pred_change_history)
        x_pc = range(pc_start_step, steps)

        plt.figure(figsize=(10, 5))
        plt.plot(x_pc, pred_change_history, 'g-', linewidth=2, marker='s', markersize=5)
        plt.xlabel('Diffusion Step')
        plt.ylabel('pred_change')
        plt.title('EasyCache Per-Step Error Signal  pred_change_t = k_{t-1} · (||x_t − x_{t-1}|| / ||v_{t-1}||)\n'
                  '(baseline run — this is what gets accumulated toward the threshold)')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(max(1, steps // 10)))
        plot_path = os.path.join(save_path, 'pred_change_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'pred_change plot saved to: {plot_path}')

        with open(os.path.join(save_path, 'pred_change.txt'), 'w') as f:
            for v in pred_change_history:
                f.write(f"{v}\n")

    # ---- Diagnostic info ----
    with open(os.path.join(save_path, 'diagnostic_info.txt'), 'w') as f:
        f.write("EasyCache Profiling Baseline\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mode: no caching — full compute every step\n")
        f.write(f"\n=== Timing ===\n")
        f.write(f"End-to-end time:   {e2e_time:.2f}s\n")
        f.write(f"DiT-only time:     {transformer.total_time:.2f}s\n")
        f.write(f"Inference steps:   {steps}\n")
        f.write(f"\n=== k_t Statistics ===\n")
        if len(k_history) > 0:
            k_arr = np.array(k_history)
            f.write(f"k_t values recorded: {len(k_arr)}  (steps {k_start_step}–{steps-1})\n")
            f.write(f"Mean:  {k_arr.mean():.4f}\n")
            f.write(f"Std:   {k_arr.std():.4f}\n")
            f.write(f"Min:   {k_arr.min():.4f}\n")
            f.write(f"Max:   {k_arr.max():.4f}\n")
        f.write(f"\n=== Prompt ===\n{outputs['prompts'][0]}\n")
        f.write(f"\n=== Seed ===\n{outputs['seeds'][0]}\n")
    logger.info(f'Diagnostic info saved to: {os.path.join(save_path, "diagnostic_info.txt")}')

    # ---- Time cost JSON ----
    time_cost = {
        "GPU_Device": torch.cuda.get_device_name(0),
        "number_prompt": 1,
        "avg_cost_time": transformer.total_time
    }
    print(f"GPU_Device: {time_cost['GPU_Device']}, avg_cost_time: {time_cost['avg_cost_time']:.2f}s")

    try:
        with open(os.path.join(args.save_path, 'time_cost.json'), "a+") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            f.seek(0)
            try:
                existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
            existing_data.append(time_cost)
            f.seek(0)
            f.truncate()
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        print(f"Error writing time cost to file: {e}")


if __name__ == "__main__":
    main()
