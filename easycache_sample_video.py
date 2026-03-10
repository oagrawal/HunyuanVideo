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

from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens
from typing import Optional, Union, Dict
import torch
import json
import numpy as np
import portalocker
import matplotlib.pyplot as plt


def easycache_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    EasyCache forward with real skipping:
    - Uses k_t to predict per-step relative output change (pred_change).
    - Accumulates pred_change and skips while accumulated < thresh.
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

    vec = self.time_in(t)
    vec = vec + self.vector_in(text_states_2)

    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(guidance)

    # Decide whether to compute or skip
    if self.cnt < self.ret_steps or self.cnt >= self.num_steps - 1:
        should_calc = True
        self.accumulated_error = 0.0
    else:
        if self.previous_raw_input is not None and self.previous_output is not None and self.k is not None:
            raw_input_change = (raw_input - self.previous_raw_input).abs().mean()
            output_norm = self.previous_output.abs().mean()
            pred_change = (self.k * (raw_input_change / output_norm)).item()

            self.pred_change_history.append(pred_change)

            self.accumulated_error += pred_change
            self.accumulated_error_history.append(self.accumulated_error)

            # Adaptive: use thresh_low for first_steps and last_steps, thresh_high for middle
            current_thresh = self.thresh
            if getattr(self, 'easycache_adaptive', False):
                fs = getattr(self, 'first_steps', 8)
                ls = getattr(self, 'last_steps', 6)
                if self.cnt <= fs or self.cnt >= self.num_steps - ls - 1:
                    current_thresh = self.thresh_low
                else:
                    current_thresh = self.thresh_high

            if self.accumulated_error < current_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_error = 0.0
        else:
            should_calc = True

    self.previous_raw_input = raw_input.clone()

    # Skip path: reuse cached residual
    if not should_calc and self.cache is not None:
        result = raw_input + self.cache
        self.cnt += 1

        torch.cuda.synchronize()
        end_time = time.time()
        self.total_time += (end_time - start_time)

        if return_dict:
            out["x"] = result
            return out
        return result

    # Full model compute
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
        img, txt = block(
            img,
            txt,
            vec,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            freqs_cis,
        )

    x = torch.cat((img, txt), 1)
    if len(self.single_blocks) > 0:
        for _, block in enumerate(self.single_blocks):
            x = block(
                x,
                vec,
                txt_seq_len,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                (freqs_cos, freqs_sin),
            )

    img = x[:, :img_seq_len, ...]
    img = self.final_layer(img, vec)
    result = self.unpatchify(img, tt, th, tw)

    # Update EasyCache state
    if self.previous_output is not None and self.prev_prev_raw_input is not None:
        output_change = (result - self.previous_output).abs().mean()
        input_change = (self.previous_raw_input - self.prev_prev_raw_input).abs().mean()
        if input_change > 0:
            self.k = output_change / input_change
            self.k_history.append(self.k.item())

    self.cache = result - raw_input
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


def easycache_baseline_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Baseline (no-caching) forward that records k_t and per-step pred_change.
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

    vec = self.time_in(t)
    vec = vec + self.vector_in(text_states_2)

    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(guidance)

    if self.previous_raw_input is not None and self.previous_output is not None and self.k is not None:
        raw_input_change = (raw_input - self.previous_raw_input).abs().mean()
        output_norm = self.previous_output.abs().mean()
        pred_change = (self.k * (raw_input_change / output_norm)).item()
        self.pred_change_history.append(pred_change)

    self.previous_raw_input = raw_input.clone()

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
        img, txt = block(
            img,
            txt,
            vec,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            freqs_cis,
        )

    x = torch.cat((img, txt), 1)
    if len(self.single_blocks) > 0:
        for _, block in enumerate(self.single_blocks):
            x = block(
                x,
                vec,
                txt_seq_len,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                (freqs_cos, freqs_sin),
            )

    img = x[:, :img_seq_len, ...]
    img = self.final_layer(img, vec)
    result = self.unpatchify(img, tt, th, tw)

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
    import argparse
    import sys

    # Add EasyCache experiment args without touching hyvideo's parser
    exp_parser = argparse.ArgumentParser(add_help=False)
    exp_parser.add_argument(
        "--easycache-mode",
        type=str,
        default="baseline",
        choices=["baseline", "easycache", "adaptive"],
        help="baseline: no skipping; easycache: fixed threshold; adaptive: low at start/end, high in middle.",
    )
    exp_parser.add_argument(
        "--easycache-thresh",
        type=float,
        default=0.025,
        help="Fixed threshold (used when mode=easycache).",
    )
    exp_parser.add_argument(
        "--easycache-thresh-low",
        type=float,
        default=0.025,
        help="Low threshold (used in adaptive mode for first_steps and last_steps).",
    )
    exp_parser.add_argument(
        "--easycache-thresh-high",
        type=float,
        default=0.050,
        help="High threshold (used in adaptive mode for middle steps).",
    )
    exp_parser.add_argument(
        "--easycache-first-steps",
        type=int,
        default=8,
        help="Steps 5..first_steps use low threshold in adaptive mode.",
    )
    exp_parser.add_argument(
        "--easycache-last-steps",
        type=int,
        default=6,
        help="Last last_steps steps use low threshold in adaptive mode.",
    )
    exp_parser.add_argument(
        "--easycache-ret-steps",
        type=int,
        default=5,
        help="Steps 0..ret_steps-1 always compute (no skipping).",
    )
    exp_args, remaining_argv = exp_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_argv
    args = parse_args()

    print(args)
    print(
        f"\n=== EasyCache Config ===\n"
        f"mode      : {exp_args.easycache_mode}\n"
        f"thresh    : {exp_args.easycache_thresh}\n"
        f"ret_steps : {exp_args.easycache_ret_steps}\n"
        f"========================\n"
    )

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    os.makedirs(args.save_path, exist_ok=True)

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args

    transformer = hunyuan_video_sampler.pipeline.transformer

    # Shared state
    transformer.__class__.cnt = 0
    transformer.__class__.num_steps = args.infer_steps
    transformer.__class__.total_time = 0.0
    transformer.__class__.k = None
    transformer.__class__.previous_raw_input = None
    transformer.__class__.previous_output = None
    transformer.__class__.prev_prev_raw_input = None
    transformer.__class__.k_history = []
    transformer.__class__.pred_change_history = []

    if exp_args.easycache_mode in ("easycache", "adaptive"):
        transformer.__class__.forward = easycache_forward
        transformer.__class__.cache = None
        transformer.__class__.accumulated_error = 0.0
        transformer.__class__.accumulated_error_history = []
        transformer.__class__.ret_steps = exp_args.easycache_ret_steps
        if exp_args.easycache_mode == "adaptive":
            transformer.__class__.easycache_adaptive = True
            transformer.__class__.thresh = exp_args.easycache_thresh_low
            transformer.__class__.thresh_low = exp_args.easycache_thresh_low
            transformer.__class__.thresh_high = exp_args.easycache_thresh_high
            transformer.__class__.first_steps = exp_args.easycache_first_steps
            transformer.__class__.last_steps = exp_args.easycache_last_steps
            mode_tag = f"easycache_adaptive_l{exp_args.easycache_thresh_low}_h{exp_args.easycache_thresh_high}"
        else:
            transformer.__class__.easycache_adaptive = False
            transformer.__class__.thresh = exp_args.easycache_thresh
            transformer.__class__.thresh_low = exp_args.easycache_thresh
            transformer.__class__.thresh_high = exp_args.easycache_thresh
            mode_tag = f"easycache_thr{exp_args.easycache_thresh}"
    else:
        transformer.__class__.forward = easycache_baseline_forward
        transformer.__class__.accumulated_error_history = []
        mode_tag = "baseline_profile"

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
        embedded_guidance_scale=args.embedded_cfg_scale,
    )
    end_time = time.time()
    e2e_time = end_time - start_time
    logger.info(f"End-to-end generation time: {e2e_time:.2f} seconds")

    samples = outputs["samples"]

    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    seed = outputs["seeds"][0]
    prompt_short = outputs["prompts"][0][:50].replace("/", "").replace(" ", "_")
    generation_folder = f"{mode_tag}_{time_flag}_seed{seed}_{prompt_short}"
    save_path = os.path.join(args.save_path, generation_folder)
    os.makedirs(save_path, exist_ok=True)

    if "LOCAL_RANK" not in os.environ or int(os.environ.get("LOCAL_RANK", 0)) == 0:
        for i, sample in enumerate(samples):
            video_path = os.path.join(save_path, "video.mp4")
            save_videos_grid(samples[i].unsqueeze(0), video_path, fps=24)
            logger.info(f"Sample saved to: {video_path}")

    k_history = transformer.k_history
    pred_change_history = transformer.pred_change_history
    acc_error_history = transformer.accumulated_error_history
    steps = args.infer_steps

    # k_t plot
    if len(k_history) > 0:
        k_start_step = steps - len(k_history)
        x_k = range(k_start_step, steps)

        plt.figure(figsize=(10, 5))
        plt.plot(x_k, k_history, "b-", linewidth=2, marker="o", markersize=5)
        plt.xlabel("Diffusion Step")
        plt.ylabel("k_t")
        plt.title(
            "EasyCache Transformation Rate  k_t = ||v_t − v_{t-1}|| / ||x_t − x_{t-1}||"
        )
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(max(1, steps // 10)))
        plot_path = os.path.join(save_path, "k_t_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"k_t plot saved to: {plot_path}")

        with open(os.path.join(save_path, "k_t.txt"), "w") as f:
            for v in k_history:
                f.write(f"{v}\n")

    # pred_change plot
    if len(pred_change_history) > 0:
        pc_start_step = steps - len(pred_change_history)
        x_pc = range(pc_start_step, steps)

        plt.figure(figsize=(10, 5))
        plt.plot(x_pc, pred_change_history, "g-", linewidth=2, marker="s", markersize=5)
        plt.xlabel("Diffusion Step")
        plt.ylabel("pred_change")
        plt.title(
            "EasyCache Per-Step Relative Change  "
            "pred_change_t = k_{t-1} · (||x_t − x_{t-1}|| / ||v_{t-1}||)"
        )
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(max(1, steps // 10)))
        plot_path = os.path.join(save_path, "pred_change_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"pred_change plot saved to: {plot_path}")

        with open(os.path.join(save_path, "pred_change.txt"), "w") as f:
            for v in pred_change_history:
                f.write(f"{v}\n")

    # Accumulated EasyCache metric (only meaningful when skipping)
    if len(acc_error_history) > 0:
        ae_start_step = steps - len(acc_error_history)
        x_ae = range(ae_start_step, steps)

        plt.figure(figsize=(10, 5))
        plt.plot(x_ae, acc_error_history, "r-", linewidth=2, marker="^", markersize=5)
        plt.xlabel("Diffusion Step")
        plt.ylabel("Accumulated pred_change")
        plt.title(
            "EasyCache Accumulated Metric Over Steps\n"
            "(sawtooth pattern: grows while skipping, resets when model runs)"
        )
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(max(1, steps // 10)))
        plot_path = os.path.join(save_path, "acc_pred_change_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Accumulated pred_change plot saved to: {plot_path}")

        with open(os.path.join(save_path, "acc_pred_change.txt"), "w") as f:
            for v in acc_error_history:
                f.write(f"{v}\n")

    diagnostic_path = os.path.join(save_path, "diagnostic_info.txt")
    with open(diagnostic_path, "w") as f:
        f.write("EasyCache Experiment\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mode: {exp_args.easycache_mode}\n")
        f.write(f"Threshold: {exp_args.easycache_thresh}\n")
        f.write(f"ret_steps: {exp_args.easycache_ret_steps}\n")
        f.write("\n=== Timing ===\n")
        f.write(f"End-to-end time:   {e2e_time:.2f}s\n")
        f.write(f"DiT-only time:     {transformer.total_time:.2f}s\n")
        f.write(f"Inference steps:   {steps}\n")
        if len(k_history) > 0:
            k_arr = np.array(k_history)
            f.write("\n=== k_t Statistics ===\n")
            f.write(f"k_t values recorded: {len(k_arr)}\n")
            f.write(f"Mean:  {k_arr.mean():.4f}\n")
            f.write(f"Std:   {k_arr.std():.4f}\n")
            f.write(f"Min:   {k_arr.min():.4f}\n")
            f.write(f"Max:   {k_arr.max():.4f}\n")
        f.write("\n=== Prompt ===\n")
        f.write(f"{outputs['prompts'][0]}\n")
        f.write("\n=== Seed ===\n")
        f.write(f"{outputs['seeds'][0]}\n")
    logger.info(f"Diagnostic info saved to: {diagnostic_path}")

    time_cost = {
        "GPU_Device": torch.cuda.get_device_name(0),
        "number_prompt": 1,
        "avg_cost_time": transformer.total_time,
    }
    print(
        f"GPU_Device: {time_cost['GPU_Device']}, "
        f"avg_cost_time: {time_cost['avg_cost_time']:.2f}s"
    )

    try:
        json_path = os.path.join(args.save_path, "time_cost.json")
        with open(json_path, "a+") as f:
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
