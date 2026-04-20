#!/usr/bin/env python3
"""Run VBench evaluation on EasyCache generated videos."""

# Compatibility: VBench's "scene" (and possibly "motion_smoothness") dimension imports
# from transformers.modeling_utils, but in newer transformers these were moved or removed.
# Patch modeling_utils before importing vbench.
def _patch_torch_load():
    """PyTorch 2.6 changed torch.load default weights_only from False to True.
    VBench loads checkpoints (e.g. AMT for motion_smoothness) without the flag,
    so we restore the pre-2.6 behaviour globally for this process."""
    import torch
    import functools
    _original_load = torch.load
    @functools.wraps(_original_load)
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)
    torch.load = _patched_load


_patch_torch_load()


def _patch_transformers_modeling_utils():
    import transformers.modeling_utils as _mutils
    # apply_chunking_to_forward: moved to pytorch_utils
    if not hasattr(_mutils, "apply_chunking_to_forward"):
        try:
            from transformers.pytorch_utils import apply_chunking_to_forward
            _mutils.apply_chunking_to_forward = apply_chunking_to_forward
        except ImportError:
            pass
    # prune_linear_layer: moved to pytorch_utils
    if not hasattr(_mutils, "prune_linear_layer"):
        try:
            from transformers.pytorch_utils import prune_linear_layer
            _mutils.prune_linear_layer = prune_linear_layer
        except ImportError:
            pass
    # find_pruneable_heads_and_indices: removed in some transformers; provide minimal impl
    if not hasattr(_mutils, "find_pruneable_heads_and_indices"):
        try:
            from transformers.pytorch_utils import find_pruneable_heads_and_indices
            _mutils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        except ImportError:
            import torch
            def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                from typing import List, Set, Tuple
                heads = set(heads) - already_pruned_heads
                if not heads:
                    return set(), torch.arange(0, n_heads * head_size, device="cpu")
                mask = torch.ones(n_heads, head_size)
                for h in heads:
                    mask[h] = 0
                index = torch.where(mask.view(-1) == 1)[0]
                return heads, index
            _mutils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    return _mutils


_patch_transformers_modeling_utils()

import argparse
import json
import os
import sys
import torch
from vbench import VBench

# All 16 VBench dimensions; evaluation must produce one _eval_results.json per dimension.
DIMENSIONS = [
    "subject_consistency", "imaging_quality", "background_consistency",
    "motion_smoothness", "overall_consistency", "human_action",
    "multiple_objects", "spatial_relationship", "object_class", "color",
    "aesthetic_quality", "appearance_style", "temporal_flickering",
    "scene", "temporal_style", "dynamic_degree",
]
assert len(DIMENSIONS) == 16, "VBench expects exactly 16 dimensions"


def _dimension_has_valid_result(save_path: str, dimension: str) -> bool:
    """True if dimension has eval_results.json with a non-NaN score."""
    p = os.path.join(save_path, f"{dimension}_eval_results.json")
    if not os.path.isfile(p):
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        val = data.get(dimension)
        if not isinstance(val, list) or len(val) < 1:
            return False
        score = val[0]
        if score is None or (isinstance(score, float) and score != score):
            return False
        return True
    except Exception:
        return False


ALL_MODES = [
    "easycache_baseline",
    "easycache_fixed_0.025",
    "easycache_fixed_0.050",
    "easycache_adaptive",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", default="vbench_eval_easycache/videos")
    p.add_argument("--save-dir", default="vbench_eval_easycache/vbench_scores")
    p.add_argument("--full-info", default="vbench_eval/prompts_subset.json")
    p.add_argument("--modes", default="all")
    p.add_argument("--dimensions", default="all")
    args = p.parse_args()

    modes = ALL_MODES if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    dims = DIMENSIONS if args.dimensions == "all" else [d.strip() for d in args.dimensions.split(",")]

    print("=" * 70)
    print("EasyCache VBench Evaluation")
    print(f"Video dir: {args.video_dir}  Save dir: {args.save_dir}")
    print(f"Modes: {modes}  Dimensions: {len(dims)}")
    print("=" * 70)

    for mode in modes:
        video_path = os.path.join(args.video_dir, mode)
        if not os.path.exists(video_path):
            print(f"WARNING: {video_path} not found, skipping")
            continue
        save_path = os.path.join(args.save_dir, mode)
        os.makedirs(save_path, exist_ok=True)

        remaining = [d for d in dims if not _dimension_has_valid_result(save_path, d)]
        if not remaining:
            print(f"  {mode}: all 16 dimensions done")
            continue

        print(f"\nEvaluating {mode} ({len(remaining)} dims)...")
        for i, dimension in enumerate(remaining, 1):
            print(f"  [{i}/{len(remaining)}] {dimension}...")
            full_info_path = os.path.join(save_path, f"{dimension}_full_info.json")
            eval_path = os.path.join(save_path, f"{dimension}_eval_results.json")
            if os.path.isfile(full_info_path) and not os.path.isfile(eval_path):
                try:
                    os.remove(full_info_path)
                    print(f"    (removed stale {dimension}_full_info.json to regenerate with current videos_path)")
                except OSError:
                    pass
            try:
                vbench = VBench(torch.device("cuda"), args.full_info, save_path)
                vbench.evaluate(
                    videos_path=video_path,
                    name=dimension,
                    local=False,
                    read_frame=False,
                    dimension_list=[dimension],
                    mode="vbench_standard",
                    imaging_quality_preprocessing_mode="longer",
                )
            except Exception as e:
                print(f"    ERROR: {e}")

    # Verify all 16 dimensions have eval results for each evaluated mode
    print("\n" + "=" * 70)
    print("Verification: 16 dimensions per mode")
    print("=" * 70)
    all_ok = True
    for mode in modes:
        save_path = os.path.join(args.save_dir, mode)
        if not os.path.isdir(save_path):
            continue
        missing = []
        empty_or_nan = []
        for dim in DIMENSIONS:
            p = os.path.join(save_path, f"{dim}_eval_results.json")
            if not os.path.isfile(p):
                missing.append(dim)
                all_ok = False
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
                val = data.get(dim)
                if not isinstance(val, list) or len(val) < 1:
                    empty_or_nan.append(dim)
                    all_ok = False
                else:
                    score = val[0]
                    if score is None or (isinstance(score, float) and score != score):
                        empty_or_nan.append(dim)
                        all_ok = False
            except Exception:
                empty_or_nan.append(dim)
                all_ok = False
        if missing or empty_or_nan:
            print(f"  {mode}: MISSING dims: {missing or 'none'}; empty/NaN: {empty_or_nan or 'none'}")
        else:
            print(f"  {mode}: OK (16/16 dimensions)")
    if all_ok:
        print("All modes have all 16 dimensions evaluated.")
    else:
        print("WARNING: Some dimensions are missing or empty. Re-run with --dimensions <missing> to retry.")
    print("\nEasyCache VBench evaluation complete.")


if __name__ == "__main__":
    main()
