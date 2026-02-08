#!/usr/bin/env python3
"""
Select a representative subset of VBench prompts for evaluation.

VBench has 946 prompts across 16 dimensions, but many prompts share dimensions
in groups. This script selects N prompts per dimension group to create a smaller
but complete subset that covers all 16 dimensions.

Dimension groups (prompts are shared within a group):
  Group A: subject_consistency + dynamic_degree + motion_smoothness (72 prompts)
  Group B: overall_consistency + aesthetic_quality + imaging_quality (93 prompts)
  Group C: scene + background_consistency (86 prompts)
  Group D: temporal_flickering (75 prompts, standalone)
  Group E: human_action (100 prompts, standalone)
  Group F: temporal_style (100 prompts, standalone)
  Group G: appearance_style (90 prompts, standalone)
  Group H: color (85 prompts, standalone)
  Group I: spatial_relationship (84 prompts, standalone)
  Group J: multiple_objects (82 prompts, standalone)
  Group K: object_class (79 prompts, standalone)

With 5 prompts per group: 11 groups × 5 = 55 unique prompts covering all 16 dims.
With 10 prompts per group: 11 groups × 10 = 110 unique prompts.

Usage:
    python select_prompts.py                          # 5 per group (default)
    python select_prompts.py --per-group 10           # 10 per group
    python select_prompts.py --seed 42                # different random selection
    python select_prompts.py --strategy first         # take first N instead of random
"""

import json
import argparse
import random
from collections import defaultdict
from pathlib import Path


# Define dimension groups (dimensions that always appear together on the same prompts)
DIMENSION_GROUPS = {
    "A_subject_motion_dynamic": frozenset(["subject_consistency", "dynamic_degree", "motion_smoothness"]),
    "B_overall_aesthetic_imaging": frozenset(["overall_consistency", "aesthetic_quality", "imaging_quality"]),
    "C_scene_background": frozenset(["scene", "background_consistency"]),
    "D_temporal_flickering": frozenset(["temporal_flickering"]),
    "E_human_action": frozenset(["human_action"]),
    "F_temporal_style": frozenset(["temporal_style"]),
    "G_appearance_style": frozenset(["appearance_style"]),
    "H_color": frozenset(["color"]),
    "I_spatial_relationship": frozenset(["spatial_relationship"]),
    "J_multiple_objects": frozenset(["multiple_objects"]),
    "K_object_class": frozenset(["object_class"]),
}


def identify_group(dimensions):
    """Identify which group a prompt belongs to based on its dimension list."""
    dim_set = frozenset(dimensions)
    for group_name, group_dims in DIMENSION_GROUPS.items():
        if dim_set == group_dims:
            return group_name
    return None


def select_prompts(input_path, per_group=5, strategy="first", seed=42):
    """Select a subset of prompts from VBench_full_info.json."""
    
    with open(input_path, "r") as f:
        full_info = json.load(f)
    
    print(f"Loaded {len(full_info)} prompts from {input_path}")
    
    # Group prompts by their dimension group
    grouped = defaultdict(list)
    ungrouped = []
    
    for entry in full_info:
        group = identify_group(entry["dimension"])
        if group:
            grouped[group].append(entry)
        else:
            ungrouped.append(entry)
    
    if ungrouped:
        print(f"\nWARNING: {len(ungrouped)} prompts didn't match any group:")
        for entry in ungrouped[:5]:
            print(f"  dims={entry['dimension']}, prompt={entry['prompt_en'][:60]}")
    
    # Print group sizes
    print(f"\nDimension groups:")
    total_selected = 0
    for group_name in sorted(grouped.keys()):
        prompts = grouped[group_name]
        dims = list(DIMENSION_GROUPS[group_name])
        n_select = min(per_group, len(prompts))
        total_selected += n_select
        print(f"  {group_name}: {len(prompts)} available, selecting {n_select}")
        print(f"    Dimensions: {', '.join(sorted(dims))}")
    
    print(f"\nTotal unique prompts to select: {total_selected}")
    
    # Select prompts from each group
    selected = []
    
    for group_name in sorted(grouped.keys()):
        prompts = grouped[group_name]
        n_select = min(per_group, len(prompts))
        
        if strategy == "first":
            chosen = prompts[:n_select]
        elif strategy == "random":
            rng = random.Random(seed)
            chosen = rng.sample(prompts, n_select)
        elif strategy == "evenly_spaced":
            # Take evenly spaced prompts from the list
            step = len(prompts) / n_select
            indices = [int(i * step) for i in range(n_select)]
            chosen = [prompts[i] for i in indices]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        selected.extend(chosen)
    
    # Verify dimension coverage
    covered_dims = set()
    for entry in selected:
        covered_dims.update(entry["dimension"])
    
    all_dims = set()
    for group_dims in DIMENSION_GROUPS.values():
        all_dims.update(group_dims)
    
    missing = all_dims - covered_dims
    if missing:
        print(f"\nERROR: Missing dimensions: {missing}")
    else:
        print(f"\nAll 16 dimensions covered.")
    
    # Print per-dimension coverage
    dim_counts = defaultdict(int)
    for entry in selected:
        for dim in entry["dimension"]:
            dim_counts[dim] += 1
    
    print(f"\nPer-dimension prompt counts:")
    for dim in sorted(dim_counts.keys()):
        print(f"  {dim}: {dim_counts[dim]}")
    
    return selected


def main():
    parser = argparse.ArgumentParser(description="Select VBench prompt subset")
    parser.add_argument("--input", type=str, 
                        default=str(Path(__file__).parent.parent / "teacache_eval" / "teacache" / "vbench" / "VBench_full_info.json"),
                        help="Path to VBench_full_info.json")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "prompts_subset.json"),
                        help="Output path for selected prompts")
    parser.add_argument("--per-group", type=int, default=5,
                        help="Number of prompts to select per dimension group (default: 5)")
    parser.add_argument("--strategy", type=str, default="first",
                        choices=["first", "random", "evenly_spaced"],
                        help="Selection strategy (default: first)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for 'random' strategy")
    args = parser.parse_args()
    
    selected = select_prompts(args.input, args.per_group, args.strategy, args.seed)
    
    # Save selected prompts
    with open(args.output, "w") as f:
        json.dump(selected, f, indent=4)
    
    print(f"\nSaved {len(selected)} prompts to {args.output}")
    
    # Also print all selected prompts for review
    print(f"\n{'='*80}")
    print("Selected prompts:")
    print(f"{'='*80}")
    for i, entry in enumerate(selected):
        dims = ", ".join(entry["dimension"])
        print(f"  {i+1:3d}. [{dims}]")
        print(f"       {entry['prompt_en']}")


if __name__ == "__main__":
    main()
