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

# Wan 2.1 modes (for --modes wan_baseline,wan_fixed_0.1,...)
WAN_MODES = [
    "wan_baseline",
    "wan_fixed_0.1",
    "wan_fixed_0.2",
    "wan_adaptive",
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


def load_generation_logs(log_dir):
    """Load generation timing data from all generation_log_*.json files.
    Merges multiple log files (from multi-GPU runs) into one."""
    import glob

    mode_times = {}

    # Find all generation log files
    patterns = [
        os.path.join(log_dir, "generation_log_*.json"),
        os.path.join(log_dir, "generation_log.json"),  # legacy single-file format
    ]
    log_files = []
    for pattern in patterns:
        log_files.extend(glob.glob(pattern))
    log_files = sorted(set(log_files))

    if not log_files:
        return {}

    for log_path in log_files:
        with open(log_path, "r") as f:
            log = json.load(f)
        for run in log.get("runs", []):
            if "time_seconds" not in run:
                continue
            mode = run["mode"]
            if mode not in mode_times:
                mode_times[mode] = []
            mode_times[mode].append(run["time_seconds"])

    if not mode_times:
        return {}

    print(f"  Loaded timing data from {len(log_files)} log file(s): {[os.path.basename(f) for f in log_files]}")

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
    parser.add_argument("--gen-log-dir", type=str, default="vbench_eval/videos",
                        help="Directory containing generation_log_*.json files")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="(deprecated, CSVs are always saved to vbench_eval/)")
    parser.add_argument("--output-json", type=str, default="vbench_eval/all_comparison_results.json",
                        help="JSON output path")
    parser.add_argument("--modes", type=str, default=None,
                        help="Comma-separated mode names (e.g. wan_baseline,wan_fixed_0.1,wan_fixed_0.2,wan_adaptive). If not set, uses HunyuanVideo modes.")
    args = parser.parse_args()

    # Resolve mode list (for Wan use --modes wan_baseline,wan_fixed_0.1,wan_fixed_0.2,wan_adaptive)
    if args.modes is not None:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        modes = ALL_MODES

    print("=" * 90)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 90)

    # ---- Load all data ----
    timing = load_generation_logs(args.gen_log_dir)
    fidelity = load_fidelity_metrics(args.fidelity_dir)

    all_data = {}
    for mode in modes:
        score_dir = os.path.join(args.scores_dir, mode)
        raw_scores = load_vbench_scores(score_dir)
        aggregate = compute_vbench_aggregate(raw_scores)
        all_data[mode] = {
            "raw_vbench": raw_scores,
            "aggregate": aggregate,
            "timing": timing.get(mode),
            "fidelity": fidelity.get(mode),
        }

    # ---- Shared labels and baseline time ----
    MODE_LABELS = {
        "hunyuan_baseline": "HunyuanVideo (no cache)",
        "hunyuan_fixed_0.1": "TeaCache 0.1",
        "hunyuan_fixed_0.2": "TeaCache 0.2",
        "hunyuan_adaptive": "TeaCache Adaptive",
        "wan_baseline": "Wan 2.1 (no cache)",
        "wan_fixed_0.1": "TeaCache 0.1",
        "wan_fixed_0.2": "TeaCache 0.2",
        "wan_adaptive": "TeaCache Adaptive",
        "mochi_baseline": "Mochi (baseline)",
    }
    # Baseline for speedup: first mode is treated as baseline if no *_baseline in timing
    baseline_time = (
        timing.get("wan_baseline", {}).get("avg_time")
        or timing.get("hunyuan_baseline", {}).get("avg_time")
        or (timing.get(modes[0], {}).get("avg_time") if timing else None)
    )

    # Precompute per-mode latency string
    def get_latency(mode):
        t = all_data[mode].get("timing")
        if t:
            return f"{t['avg_time']:.0f}s"
        return "—"

    # Ordered dimensions: quality first, then semantic
    QUALITY_DIMS_ORDERED = [
        "subject_consistency", "background_consistency", "temporal_flickering",
        "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    ]
    SEMANTIC_DIMS_ORDERED = [
        "object_class", "multiple_objects", "human_action", "color",
        "spatial_relationship", "scene", "appearance_style", "temporal_style",
        "overall_consistency",
    ]

    # ---- Table 1: VBench Scores (rows = modes, cols = dimensions + latency) ----
    modes_with_scores = [m for m in modes if all_data[m]["raw_vbench"]]
    if modes_with_scores:
        print(f"\n{'='*90}")
        print("TABLE 1: VBENCH SCORES (higher = better)")
        print(f"{'='*90}")

        all_dim_cols = QUALITY_DIMS_ORDERED + SEMANTIC_DIMS_ORDERED
        # Short column names for display
        short_dim = {
            "subject_consistency": "SubjCon",
            "background_consistency": "BgCon",
            "temporal_flickering": "TmpFlk",
            "motion_smoothness": "MotSmth",
            "dynamic_degree": "DynDeg",
            "aesthetic_quality": "Aesth",
            "imaging_quality": "ImgQl",
            "object_class": "ObjCls",
            "multiple_objects": "MulObj",
            "human_action": "HumAct",
            "color": "Color",
            "spatial_relationship": "SpatRl",
            "scene": "Scene",
            "appearance_style": "AppSty",
            "temporal_style": "TmpSty",
            "overall_consistency": "OvrlCn",
        }

        # Header
        header = f"{'Mode':<24}"
        for dim in QUALITY_DIMS_ORDERED:
            header += f" {short_dim[dim]:>7}"
        header += "  |"
        for dim in SEMANTIC_DIMS_ORDERED:
            header += f" {short_dim[dim]:>7}"
        header += f" {'Latency':>10}"
        print(header)

        sep = f"{'':.<24}"
        sep += " " + ".......".join(["" for _ in range(len(QUALITY_DIMS_ORDERED) + 1)])
        print("-" * len(header))

        # Label row for grouping
        quality_width = 8 * len(QUALITY_DIMS_ORDERED)
        semantic_width = 8 * len(SEMANTIC_DIMS_ORDERED)
        group_row = f"{'':24}"
        group_row += f" {'--- Quality Dimensions ---':^{quality_width}}"
        group_row += "  |"
        group_row += f" {'--- Semantic Dimensions ---':^{semantic_width}}"
        print(group_row)
        print("-" * len(header))

        # Data rows
        for mode in modes_with_scores:
            label = MODE_LABELS.get(mode, mode)
            row = f"{label:<24}"
            for dim in QUALITY_DIMS_ORDERED:
                val = all_data[mode]["raw_vbench"].get(dim)
                row += f" {float(val):>7.4f}" if val is not None else f" {'N/A':>7}"
            row += "  |"
            for dim in SEMANTIC_DIMS_ORDERED:
                val = all_data[mode]["raw_vbench"].get(dim)
                row += f" {float(val):>7.4f}" if val is not None else f" {'N/A':>7}"
            row += f" {get_latency(mode):>10}"
            print(row)

        # Aggregate row
        print("-" * len(header))
        agg_row = f"{'Aggregate Scores:':<24}"
        # Quality score under quality dims
        for i, dim in enumerate(QUALITY_DIMS_ORDERED):
            agg_row += f" {'':>7}"
        agg_row += "  |"
        for i, dim in enumerate(SEMANTIC_DIMS_ORDERED):
            agg_row += f" {'':>7}"
        agg_row += f" {'':>10}"
        print(agg_row)

        for score_name, score_key in [("  Quality Score", "quality_score"),
                                       ("  Semantic Score", "semantic_score"),
                                       ("  TOTAL SCORE", "total_score")]:
            row = f"{score_name:<24}"
            for mode in modes_with_scores:
                val = all_data[mode]["aggregate"].get(score_key)
                val_str = f"{val*100:.2f}%" if val is not None else "N/A"
                row += f" {val_str:>12}"
            print(row)

    # ---- Table 2: Fidelity Metrics (rows = modes, cols = PSNR/SSIM/LPIPS + latency) ----
    print(f"\n{'='*90}")
    print("TABLE 2: FIDELITY METRICS vs BASELINE")
    print("(higher PSNR/SSIM = better, lower LPIPS = better)")
    print(f"{'='*90}")

    fid_header = f"{'Mode':<26} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'Latency':>12}"
    print(fid_header)
    print("-" * 70)

    for mode in modes:
        label = MODE_LABELS.get(mode, mode)
        fid = all_data[mode].get("fidelity")

        psnr_str = f"{fid['psnr']['mean']:.4f}" if fid and "psnr" in fid else "—"
        ssim_str = f"{fid['ssim']['mean']:.4f}" if fid and "ssim" in fid else "—"
        lpips_str = f"{fid['lpips']['mean']:.4f}" if fid and "lpips" in fid else "—"
        latency = get_latency(mode)

        print(f"{label:<26} {psnr_str:>10} {ssim_str:>10} {lpips_str:>10} {latency:>12}")

    # ---- Table 3: Compact Summary ----
    print(f"\n{'='*90}")
    print("TABLE 3: COMPACT SUMMARY")
    print(f"{'='*90}")

    compact_header = f"{'Mode':<26} {'Speedup':>8} {'Latency':>12} {'VBench':>10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}"
    print(compact_header)
    print("-" * 82)

    compact_rows = []  # for CSV
    for mode in modes:
        label = MODE_LABELS.get(mode, mode)
        row_data = {"mode": label}

        # Speedup + Latency
        t = all_data[mode].get("timing")
        if t and baseline_time:
            speedup = baseline_time / t["avg_time"]
            speedup_str = f"{speedup:.2f}x"
            latency_str = f"{t['avg_time']:.0f}s"
        elif t:
            speedup_str = "1.00x"
            latency_str = f"{t['avg_time']:.0f}s"
        else:
            speedup_str = "—"
            latency_str = "—"
        row_data["speedup"] = speedup_str
        row_data["latency"] = latency_str

        # VBench Total
        total = all_data[mode]["aggregate"].get("total_score")
        if total is not None:
            vbench_str = f"{total*100:.2f}%"
        else:
            vbench_str = "—"
        row_data["vbench_total"] = vbench_str

        # Fidelity
        fid = all_data[mode].get("fidelity")
        for metric in ["psnr", "ssim", "lpips"]:
            if fid and metric in fid:
                row_data[metric] = f"{fid[metric]['mean']:.4f}"
            else:
                row_data[metric] = "—"

        compact_rows.append(row_data)
        print(f"{label:<26} {speedup_str:>8} {latency_str:>12} {vbench_str:>10} {row_data['psnr']:>8} {row_data['ssim']:>8} {row_data['lpips']:>8}")

    print(f"{'='*90}")

    # ---- Save results ----
    # Prepare serializable output
    output = {}
    for mode in modes:
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

    # ---- Save CSVs ----
    import csv
    output_base = os.path.dirname(args.output_json)

    # CSV 1: VBench scores (rows = modes, cols = quality dims | semantic dims | latency)
    vbench_csv_path = os.path.join(output_base, "vbench_scores_table.csv")
    try:
        all_dim_cols = QUALITY_DIMS_ORDERED + SEMANTIC_DIMS_ORDERED
        with open(vbench_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Mode"] + all_dim_cols + ["Quality Score", "Semantic Score", "Total Score", "Latency"]
            writer.writerow(header)
            for mode in modes:
                label = MODE_LABELS.get(mode, mode)
                row = [label]
                for dim in all_dim_cols:
                    val = all_data[mode]["raw_vbench"].get(dim, "")
                    row.append(f"{float(val):.4f}" if val != "" else "")
                for key in ["quality_score", "semantic_score", "total_score"]:
                    val = all_data[mode]["aggregate"].get(key)
                    row.append(f"{val*100:.2f}%" if val is not None else "")
                row.append(get_latency(mode))
                writer.writerow(row)
        print(f"VBench CSV saved to: {vbench_csv_path}")
    except Exception as e:
        print(f"Warning: Failed to save VBench CSV: {e}")

    # CSV 2: Fidelity metrics (rows = modes, cols = PSNR/SSIM/LPIPS + latency)
    fidelity_csv_path = os.path.join(output_base, "fidelity_table.csv")
    try:
        with open(fidelity_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Mode", "PSNR", "SSIM", "LPIPS", "Latency"])
            for mode in modes:
                label = MODE_LABELS.get(mode, mode)
                fid = all_data[mode].get("fidelity")
                psnr = f"{fid['psnr']['mean']:.4f}" if fid and "psnr" in fid else "—"
                ssim = f"{fid['ssim']['mean']:.4f}" if fid and "ssim" in fid else "—"
                lpips_val = f"{fid['lpips']['mean']:.4f}" if fid and "lpips" in fid else "—"
                writer.writerow([label, psnr, ssim, lpips_val, get_latency(mode)])
        print(f"Fidelity CSV saved to: {fidelity_csv_path}")
    except Exception as e:
        print(f"Warning: Failed to save fidelity CSV: {e}")

    # CSV 3: Compact summary
    summary_csv_path = os.path.join(output_base, "summary_table.csv")
    try:
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Mode", "Speedup", "Latency (s)", "VBench Total", "PSNR", "SSIM", "LPIPS"])
            for row_data in compact_rows:
                writer.writerow([
                    row_data["mode"],
                    row_data["speedup"],
                    row_data["latency"],
                    row_data["vbench_total"],
                    row_data["psnr"],
                    row_data["ssim"],
                    row_data["lpips"],
                ])
        print(f"Summary CSV saved to: {summary_csv_path}")
    except Exception as e:
        print(f"Warning: Failed to save summary CSV: {e}")

    print(f"{'='*90}")


if __name__ == "__main__":
    main()
