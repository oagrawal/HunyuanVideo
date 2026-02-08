#!/usr/bin/env python3
"""
Aggregate and display all evaluation results.

Combines VBench scores (16 dimensions) and fidelity metrics (PSNR/SSIM/LPIPS)
across all modes into a single comparison table.

Also computes speedup ratios from the generation log.

This script has NO heavy dependencies — runs outside Docker.

Usage:
    python3 vbench_eval/compare_results.py
    python3 vbench_eval/compare_results.py --output-csv results_table.csv
"""

import argparse
import json
import os
import sys

# ---- VBench score calculation (from TeaCache's cal_vbench.py) ----

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4

QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

ALL_MODES = [
    "hunyuan_baseline",
    "hunyuan_fixed_0.1",
    "hunyuan_fixed_0.2",
    "hunyuan_adaptive",
]


def load_vbench_scores(score_dir):
    """Load VBench evaluation results from a score directory.
    Returns dict of {dimension_name: raw_score}."""
    results = {}
    res_postfix = "_eval_results.json"

    if not os.path.exists(score_dir):
        return results

    for filename in os.listdir(score_dir):
        if filename.endswith(res_postfix):
            filepath = os.path.join(score_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            for key, val in data.items():
                results[key] = val[0] if isinstance(val, list) else val

    return results


def compute_vbench_aggregate(raw_scores):
    """Compute quality score, semantic score, and total score from raw VBench scores."""
    scaled = {}
    for key, val in raw_scores.items():
        dim = key.replace("_", " ") if "_" in key else key
        if dim not in NORMALIZE_DIC:
            continue
        norm = NORMALIZE_DIC[dim]
        scaled_score = (float(val) - norm["Min"]) / (norm["Max"] - norm["Min"])
        scaled_score *= DIM_WEIGHT[dim]
        scaled[dim] = scaled_score

    quality_dims = [d for d in QUALITY_LIST if d in scaled]
    semantic_dims = [d for d in SEMANTIC_LIST if d in scaled]

    quality_score = sum(scaled[d] for d in quality_dims) / sum(DIM_WEIGHT[d] for d in quality_dims) if quality_dims else None
    semantic_score = sum(scaled[d] for d in semantic_dims) / sum(DIM_WEIGHT[d] for d in semantic_dims) if semantic_dims else None

    total_score = None
    if quality_score is not None and semantic_score is not None:
        total_score = (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (
            QUALITY_WEIGHT + SEMANTIC_WEIGHT
        )

    return {
        "quality_score": quality_score,
        "semantic_score": semantic_score,
        "total_score": total_score,
        "scaled": scaled,
    }


def load_generation_log(log_path):
    """Load generation timing data."""
    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r") as f:
        log = json.load(f)

    # Compute per-mode average time
    mode_times = {}
    for run in log.get("runs", []):
        if "time_seconds" not in run:
            continue
        mode = run["mode"]
        if mode not in mode_times:
            mode_times[mode] = []
        mode_times[mode].append(run["time_seconds"])

    return {
        mode: {
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
            "num_videos": len(times),
        }
        for mode, times in mode_times.items()
    }


def load_fidelity_metrics(metrics_dir):
    """Load fidelity metrics (PSNR/SSIM/LPIPS) results."""
    results = {}
    combined_path = os.path.join(metrics_dir, "all_fidelity_results.json")

    if os.path.exists(combined_path):
        with open(combined_path, "r") as f:
            results = json.load(f)
    else:
        # Try individual files
        for filename in os.listdir(metrics_dir) if os.path.exists(metrics_dir) else []:
            if filename.endswith(".json") and "_vs_" in filename:
                filepath = os.path.join(metrics_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                results[data["mode"]] = data

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results across modes")
    parser.add_argument("--scores-dir", type=str, default="vbench_eval/vbench_scores",
                        help="Base directory for VBench scores")
    parser.add_argument("--fidelity-dir", type=str, default="vbench_eval/fidelity_metrics",
                        help="Directory for fidelity metrics")
    parser.add_argument("--gen-log", type=str, default="vbench_eval/videos/generation_log.json",
                        help="Path to generation log")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional CSV output path")
    parser.add_argument("--output-json", type=str, default="vbench_eval/all_comparison_results.json",
                        help="JSON output path")
    args = parser.parse_args()

    print("=" * 90)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 90)

    # ---- Load all data ----
    timing = load_generation_log(args.gen_log)
    fidelity = load_fidelity_metrics(args.fidelity_dir)

    all_data = {}
    for mode in ALL_MODES:
        score_dir = os.path.join(args.scores_dir, mode)
        raw_scores = load_vbench_scores(score_dir)
        aggregate = compute_vbench_aggregate(raw_scores)
        all_data[mode] = {
            "raw_vbench": raw_scores,
            "aggregate": aggregate,
            "timing": timing.get(mode),
            "fidelity": fidelity.get(mode),
        }

    # ---- Print VBench Scores ----
    modes_with_scores = [m for m in ALL_MODES if all_data[m]["raw_vbench"]]
    if modes_with_scores:
        print(f"\n{'='*90}")
        print("VBENCH SCORES (raw, higher = better)")
        print(f"{'='*90}")

        # Collect all dimensions
        all_dims = set()
        for mode in modes_with_scores:
            all_dims.update(all_data[mode]["raw_vbench"].keys())
        all_dims = sorted(all_dims)

        # Header
        header = f"{'Dimension':<28}"
        for mode in modes_with_scores:
            short = mode.replace("hunyuan_", "")
            header += f" {short:>14}"
        print(header)
        print("-" * (28 + 15 * len(modes_with_scores)))

        # Per-dimension scores
        for dim in all_dims:
            row = f"{dim:<28}"
            for mode in modes_with_scores:
                val = all_data[mode]["raw_vbench"].get(dim)
                if val is not None:
                    row += f" {float(val):>14.4f}"
                else:
                    row += f" {'N/A':>14}"
            print(row)

        # Aggregate scores
        print("-" * (28 + 15 * len(modes_with_scores)))
        for score_name, score_key in [("Quality Score", "quality_score"),
                                       ("Semantic Score", "semantic_score"),
                                       ("TOTAL SCORE", "total_score")]:
            row = f"{score_name:<28}"
            for mode in modes_with_scores:
                val = all_data[mode]["aggregate"].get(score_key)
                if val is not None:
                    row += f" {val*100:>13.2f}%"
                else:
                    row += f" {'N/A':>14}"
            print(row)

    # ---- Print Fidelity Metrics ----
    modes_with_fidelity = [m for m in ALL_MODES if all_data[m]["fidelity"]]
    if modes_with_fidelity:
        print(f"\n{'='*90}")
        print("FIDELITY METRICS vs BASELINE (higher PSNR/SSIM = better, lower LPIPS = better)")
        print(f"{'='*90}")

        header = f"{'Metric':<28}"
        for mode in modes_with_fidelity:
            short = mode.replace("hunyuan_", "")
            header += f" {short:>14}"
        print(header)
        print("-" * (28 + 15 * len(modes_with_fidelity)))

        for metric in ["psnr", "ssim", "lpips"]:
            row = f"{metric.upper():<28}"
            for mode in modes_with_fidelity:
                val = all_data[mode]["fidelity"].get(metric, {}).get("mean")
                if val is not None:
                    row += f" {val:>14.4f}"
                else:
                    row += f" {'N/A':>14}"
            print(row)

    # ---- Print Timing ----
    modes_with_timing = [m for m in ALL_MODES if all_data[m]["timing"]]
    if modes_with_timing:
        print(f"\n{'='*90}")
        print("GENERATION TIMING")
        print(f"{'='*90}")

        baseline_time = None
        if "hunyuan_baseline" in timing:
            baseline_time = timing["hunyuan_baseline"]["avg_time"]

        header = f"{'Metric':<28}"
        for mode in modes_with_timing:
            short = mode.replace("hunyuan_", "")
            header += f" {short:>14}"
        print(header)
        print("-" * (28 + 15 * len(modes_with_timing)))

        # Avg time
        row = f"{'Avg time (sec)':<28}"
        for mode in modes_with_timing:
            row += f" {all_data[mode]['timing']['avg_time']:>14.1f}"
        print(row)

        # Avg time in minutes
        row = f"{'Avg time (min)':<28}"
        for mode in modes_with_timing:
            row += f" {all_data[mode]['timing']['avg_time']/60:>14.1f}"
        print(row)

        # Speedup vs baseline
        if baseline_time:
            row = f"{'Speedup vs baseline':<28}"
            for mode in modes_with_timing:
                speedup = baseline_time / all_data[mode]["timing"]["avg_time"]
                row += f" {speedup:>13.2f}x"
            print(row)

        # Num videos
        row = f"{'Videos generated':<28}"
        for mode in modes_with_timing:
            row += f" {all_data[mode]['timing']['num_videos']:>14d}"
        print(row)

    print(f"\n{'='*90}")

    # ---- Save results ----
    # Prepare serializable output
    output = {}
    for mode in ALL_MODES:
        entry = {}
        if all_data[mode]["raw_vbench"]:
            entry["vbench_raw"] = {k: float(v) for k, v in all_data[mode]["raw_vbench"].items()}
            agg = all_data[mode]["aggregate"]
            entry["vbench_quality_score"] = agg["quality_score"]
            entry["vbench_semantic_score"] = agg["semantic_score"]
            entry["vbench_total_score"] = agg["total_score"]
        if all_data[mode]["fidelity"]:
            entry["fidelity"] = all_data[mode]["fidelity"]
        if all_data[mode]["timing"]:
            entry["timing"] = all_data[mode]["timing"]
        if entry:
            output[mode] = entry

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {args.output_json}")

    # CSV output
    if args.output_csv:
        try:
            import csv
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                header_row = ["metric"] + [m.replace("hunyuan_", "") for m in ALL_MODES]
                writer.writerow(header_row)

                # VBench dimensions
                all_dims = set()
                for mode in ALL_MODES:
                    all_dims.update(all_data[mode]["raw_vbench"].keys())
                for dim in sorted(all_dims):
                    row = [dim]
                    for mode in ALL_MODES:
                        val = all_data[mode]["raw_vbench"].get(dim, "")
                        row.append(f"{float(val):.4f}" if val != "" else "")
                    writer.writerow(row)

                # Aggregate scores
                for name, key in [("quality_score", "quality_score"),
                                  ("semantic_score", "semantic_score"),
                                  ("total_score", "total_score")]:
                    row = [name]
                    for mode in ALL_MODES:
                        val = all_data[mode]["aggregate"].get(key)
                        row.append(f"{val:.4f}" if val is not None else "")
                    writer.writerow(row)

                # Fidelity
                for metric in ["psnr", "ssim", "lpips"]:
                    row = [f"fidelity_{metric}"]
                    for mode in ALL_MODES:
                        fid = all_data[mode].get("fidelity")
                        val = fid.get(metric, {}).get("mean") if fid else None
                        row.append(f"{val:.4f}" if val is not None else "")
                    writer.writerow(row)

                # Timing
                row = ["avg_time_sec"]
                for mode in ALL_MODES:
                    t = all_data[mode].get("timing")
                    row.append(f"{t['avg_time']:.1f}" if t else "")
                writer.writerow(row)

            print(f"CSV saved to: {args.output_csv}")
        except Exception as e:
            print(f"Warning: Failed to save CSV: {e}")

    print(f"{'='*90}")


if __name__ == "__main__":
    main()
