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

import matplotlib.pyplot as plt




def teacache_forward(
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
        
        if self.enable_teacache:
            inp = img.clone()
            vec_ = vec.clone()
            txt_ = txt.clone()
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
            normed_inp = self.double_blocks[0].img_norm1(inp)
            modulated_inp = modulate(
                normed_inp, shift=img_mod1_shift, scale=img_mod1_scale
            )

            current_timestep = t.item() if torch.is_tensor(t) else t
            self.plot_timesteps.append(current_timestep)

            # if self.cnt <= 10 or self.cnt >= self.num_steps-5:
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                # First and last step: full diffusion
                should_calc = True
                self.accumulated_rel_l1_distance = 0

            else: 
                coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                rescale_func = np.poly1d(coefficients)
                delta = rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                self.delta_TEMNI.append(delta)
                self.accumulated_rel_l1_distance += delta

                # Use different thresholds based on step number:
                # - Steps 1-9 (first 10 minus first): low threshold
                # - Steps num_steps-5 to num_steps-2 (last 5 minus last): low threshold
                # - Middle steps: high threshold
                if self.cnt <= 10 or self.cnt >= self.num_steps - 5:
                    current_thresh = self.rel_l1_thresh_low
                else:
                    current_thresh = self.rel_l1_thresh_high

                if self.accumulated_rel_l1_distance < current_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                # should_calc = True
            
            self.acc_delta_TEMNI.append(self.accumulated_rel_l1_distance)
            self.previous_modulated_input = modulated_inp  
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0          
        
        if self.enable_teacache:
            if not should_calc:
                img += self.previous_residual
            else:
                ori_img = img.clone()
                # --------------------- Pass through DiT blocks ------------------------
                for _, block in enumerate(self.double_blocks):
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
                
                # Compute L1 distance of output state (img) between timesteps
                if self.previous_img_output is not None:
                    # Compute relative L1 distance: |current - previous| / |previous|
                    l1_distance = ((img - self.previous_img_output).abs().mean() / 
                                  self.previous_img_output.abs().mean()).cpu().item()
                    self.delta_true_output.append(l1_distance)
                else:
                    # First timestep, no previous state to compare
                    self.delta_true_output.append(0.0)
                
                self.previous_img_output = img.clone()
                
                self.previous_residual = img - ori_img
        else:        
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
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

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img


def main():
    args = parse_args()
    
    # Parse experiment-specific arguments
    import argparse
    exp_parser = argparse.ArgumentParser(add_help=False)
    exp_parser.add_argument('--teacache-mode', type=str, default='adaptive', 
                            choices=['none', 'fixed', 'adaptive'],
                            help='TeaCache mode: none (no caching), fixed (single threshold), adaptive (different thresholds for start/end vs middle)')
    exp_parser.add_argument('--fixed-thresh', type=float, default=0.10,
                            help='Fixed threshold value (used when teacache-mode=fixed)')
    exp_parser.add_argument('--thresh-low', type=float, default=0.05,
                            help='Low threshold for first 10 and last 5 steps (used when teacache-mode=adaptive)')
    exp_parser.add_argument('--thresh-high', type=float, default=0.30,
                            help='High threshold for middle steps (used when teacache-mode=adaptive)')
    exp_args, _ = exp_parser.parse_known_args()
    
    # Store experiment args
    args.teacache_mode = exp_args.teacache_mode
    args.fixed_thresh = exp_args.fixed_thresh
    args.thresh_low = exp_args.thresh_low
    args.thresh_high = exp_args.thresh_high
    
    print(args)
    print(f"\n=== Experiment Config ===")
    print(f"TeaCache mode: {args.teacache_mode}")
    if args.teacache_mode == 'fixed':
        print(f"Fixed threshold: {args.fixed_thresh}")
    elif args.teacache_mode == 'adaptive':
        print(f"Thresh low (first 10 + last 5): {args.thresh_low}")
        print(f"Thresh high (middle): {args.thresh_high}")
    print(f"=========================\n")
    
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args (preserve experiment args)
    teacache_mode = args.teacache_mode
    fixed_thresh = args.fixed_thresh
    thresh_low = args.thresh_low
    thresh_high = args.thresh_high
    args = hunyuan_video_sampler.args
    args.teacache_mode = teacache_mode
    args.fixed_thresh = fixed_thresh
    args.thresh_low = thresh_low
    args.thresh_high = thresh_high

    # TeaCache configuration based on mode
    if args.teacache_mode == 'none':
        # No caching - disable TeaCache entirely
        hunyuan_video_sampler.pipeline.transformer.__class__.enable_teacache = False
        logger.info("TeaCache DISABLED (no caching baseline)")
    else:
        # Enable TeaCache
        hunyuan_video_sampler.pipeline.transformer.__class__.enable_teacache = True
        hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
        hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
        hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
        hunyuan_video_sampler.pipeline.transformer.__class__.previous_modulated_input = None
        hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = None
        hunyuan_video_sampler.pipeline.transformer.__class__.forward = teacache_forward
        
        if args.teacache_mode == 'fixed':
            # Fixed threshold - same value for all steps
            hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh_low = args.fixed_thresh
            hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh_high = args.fixed_thresh
            logger.info(f"TeaCache FIXED threshold: {args.fixed_thresh}")
        else:
            # Adaptive threshold - different for start/end vs middle
            hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh_low = args.thresh_low
            hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh_high = args.thresh_high
            logger.info(f"TeaCache ADAPTIVE thresholds: low={args.thresh_low}, high={args.thresh_high}")
    
    # Tracking lists (always initialize for consistency)
    hunyuan_video_sampler.pipeline.transformer.__class__.acc_delta_TEMNI = []
    hunyuan_video_sampler.pipeline.transformer.__class__.plot_timesteps = []
    hunyuan_video_sampler.pipeline.transformer.__class__.delta_TEMNI = []
    hunyuan_video_sampler.pipeline.transformer.__class__.delta_true_output = []
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_img_output = None

    # Start sampling
    # TODO: batch inference check
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
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    end_time = time.time()
    e2e_time = end_time - start_time
    logger.info(f"End-to-end generation time: {e2e_time:.2f} seconds")    
    samples = outputs['samples']
    
    # Create unique folder for this generation
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    seed = outputs['seeds'][0]  # Use first seed for naming
    prompt_short = outputs['prompts'][0][:50].replace('/', '').replace(' ', '_')  # Shorter and filesystem-safe
    
    # Create experiment tag for folder naming
    if args.teacache_mode == 'none':
        exp_tag = "nocache"
    elif args.teacache_mode == 'fixed':
        exp_tag = f"fixed_{args.fixed_thresh}"
    else:  # adaptive
        exp_tag = f"adaptive_{args.thresh_low}_{args.thresh_high}"
    
    # Create generation-specific folder inside base save_path
    base_save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    generation_folder = f"{exp_tag}_{time_flag}_seed{seed}_{prompt_short}"
    save_path = os.path.join(base_save_path, generation_folder)
    os.makedirs(save_path, exist_ok=True)
    
    # Plot the accumulated delta TEMNI
    if len(hunyuan_video_sampler.pipeline.transformer.acc_delta_TEMNI) > 0:
        plt.figure(figsize=(10, 6))
        
        # Plot with individual markers at each data point
        plt.plot(range(1, len(hunyuan_video_sampler.pipeline.transformer.acc_delta_TEMNI) + 1), 
                hunyuan_video_sampler.pipeline.transformer.acc_delta_TEMNI, 
                'b-', linewidth=2, marker='o', markersize=6)
        
        plt.xlabel('Timestep Number')
        plt.ylabel('Accumulated Delta TEMNI')
        plt.title('Accumulated Delta TEMNI over Timesteps')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis ticks to increments of 10
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        
        # Save plot in generation-specific folder
        plot_path = os.path.join(save_path, 'acc_delta_TEMNI_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f'Accumulated delta TEMNI plot saved to: {plot_path}')
        
        plt.close()

    # Plot the delta TEMNI values
    if len(hunyuan_video_sampler.pipeline.transformer.delta_TEMNI) > 0:
        plt.figure(figsize=(10, 6))
        
        plt.plot(range(1, len(hunyuan_video_sampler.pipeline.transformer.delta_TEMNI) + 1), 
                hunyuan_video_sampler.pipeline.transformer.delta_TEMNI, 
                'g-', linewidth=2, marker='s', markersize=6)
        
        plt.xlabel('Step Number')
        plt.ylabel('Delta TEMNI')
        plt.title('Delta TEMNI over Steps')
        plt.grid(True, alpha=0.3)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        
        plot_path = os.path.join(save_path, 'delta_TEMNI_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f'Delta TEMNI plot saved to: {plot_path}')
        
        plt.close()

    # Plot the delta true output
    if len(hunyuan_video_sampler.pipeline.transformer.delta_true_output) > 0:
        plt.figure(figsize=(10, 6))
        
        plt.plot(range(1, len(hunyuan_video_sampler.pipeline.transformer.delta_true_output) + 1), 
                hunyuan_video_sampler.pipeline.transformer.delta_true_output, 
                'r-', linewidth=2, marker='o', markersize=6)
        
        plt.xlabel('Timestep Number')
        plt.ylabel('Delta True Output')
        plt.title('Delta True Output Between Timesteps')
        plt.grid(True, alpha=0.3)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        
        plot_path = os.path.join(save_path, 'delta_true_output_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f'Delta true output plot saved to: {plot_path}')
        
        plt.close()

    # Save delta TEMNI values to text file
    if len(hunyuan_video_sampler.pipeline.transformer.delta_TEMNI) > 0:
        delta_path = os.path.join(save_path, 'delta_TEMNI.txt')
        with open(delta_path, 'w') as f:
            for delta in hunyuan_video_sampler.pipeline.transformer.delta_TEMNI:
                f.write(f"{delta}\n")
        logger.info(f'Delta TEMNI values saved to: {delta_path}')

    # Save delta true output to text file
    if len(hunyuan_video_sampler.pipeline.transformer.delta_true_output) > 0:
        delta_true_output_path = os.path.join(save_path, 'delta_true_output.txt')
        with open(delta_true_output_path, 'w') as f:
            for delta_out in hunyuan_video_sampler.pipeline.transformer.delta_true_output:
                f.write(f"{delta_out}\n")
        logger.info(f'Delta true output saved to: {delta_true_output_path}')


    # ===================================================================
    # Always save diagnostic info for experiment tracking
    diagnostic_path = os.path.join(save_path, 'diagnostic_info.txt')
    with open(diagnostic_path, 'w') as f:
        f.write("Experiment Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"\n=== Experiment Config ===\n")
        f.write(f"TeaCache mode: {args.teacache_mode}\n")
        if args.teacache_mode == 'none':
            f.write(f"(No caching - baseline)\n")
        elif args.teacache_mode == 'fixed':
            f.write(f"Fixed threshold: {args.fixed_thresh}\n")
        elif args.teacache_mode == 'adaptive':
            f.write(f"Thresh low (first 10 + last 5): {args.thresh_low}\n")
            f.write(f"Thresh high (middle): {args.thresh_high}\n")
        f.write(f"\n=== Timing ===\n")
        f.write(f"End-to-end generation time: {e2e_time:.2f} seconds\n")
        f.write(f"Inference steps: {args.infer_steps}\n")
        f.write(f"\n=== Prompt ===\n")
        f.write(f"{outputs['prompts'][0]}\n")
        f.write(f"\n=== Seed ===\n")
        f.write(f"{outputs['seeds'][0]}\n")
    logger.info(f'Diagnostic info saved to: {diagnostic_path}')
    
    # Save raw acc_delta_TEMNI and timesteps for debugging (only when teacache enabled)
    if len(hunyuan_video_sampler.pipeline.transformer.acc_delta_TEMNI) > 0:
        # Save acc_delta_TEMNI
        acc_delta_TEMNI_path = os.path.join(save_path, 'acc_delta_TEMNI.txt')
        with open(acc_delta_TEMNI_path, 'w') as f:
            for metric in hunyuan_video_sampler.pipeline.transformer.acc_delta_TEMNI:
                f.write(f"{metric}\n")
        logger.info(f'Accumulated delta TEMNI saved to: {acc_delta_TEMNI_path}')
        
        # Save timesteps
        timesteps_path = os.path.join(save_path, 'timesteps.txt')
        with open(timesteps_path, 'w') as f:
            for timestep in hunyuan_video_sampler.pipeline.transformer.plot_timesteps:
                f.write(f"{timestep}\n")
        logger.info(f'Timesteps saved to: {timesteps_path}')
    # ===================================================================

    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            video_path = os.path.join(save_path, 'video.mp4')
            save_videos_grid(sample, video_path, fps=24)
            logger.info(f'Sample save to: {video_path}')


if __name__ == "__main__":
    main()
