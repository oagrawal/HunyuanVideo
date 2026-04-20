#!/usr/bin/env python3
"""Aggregate EasyCache VBench + fidelity + timing results. Runs outside Docker."""

import argparse
import json
import os
import glob

# Same VBench scoring as vbench_eval/compare_results.py
QUALITY_LIST = ["subject consistency", "background consistency", "temporal flickering", "motion smoothness", "aesthetic quality", "imaging quality", "dynamic degree"]
SEMANTIC_LIST = ["object class", "multiple objects", "human action", "color", "spatial relationship", "scene", "appearance style", "temporal style", "overall consistency"]
QUALITY_WEIGHT, SEMANTIC_WEIGHT = 4, 1

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0}, "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0}, "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0}, "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0}, "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0}, "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0}, "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222}, "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364}, "overall consistency": {"Min": 0.0, "Max": 0.364},
}
DIM_WEIGHT = {d: 1 for d in QUALITY_LIST + SEMANTIC_LIST}
DIM_WEIGHT["dynamic degree"] = 0.5

ALL_MODES = ["easycache_baseline", "easycache_fixed_0.025", "easycache_fixed_0.050", "easycache_adaptive"]
MODE_LABELS = {
    "easycache_baseline": "EasyCache baseline",
    "easycache_fixed_0.025": "EasyCache fixed 0.025",
    "easycache_fixed_0.050": "EasyCache fixed 0.050",
    "easycache_adaptive": "EasyCache adaptive",
}


def load_vbench_scores(score_dir):
    r = {}
    if not os.path.exists(score_dir):
        return r
    for f in os.listdir(score_dir):
        if f.endswith("_eval_results.json"):
            with open(os.path.join(score_dir, f)) as fp:
                d = json.load(fp)
            for k, v in d.items():
                r[k] = v[0] if isinstance(v, list) else v
    return r


def compute_aggregate(raw):
    scaled = {}
    for k, v in raw.items():
        dim = k.replace("_", " ")
        if dim in NORMALIZE_DIC:
            n = NORMALIZE_DIC[dim]
            s = (float(v) - n["Min"]) / (n["Max"] - n["Min"])
            scaled[dim] = s * DIM_WEIGHT.get(dim, 1)
    q = [scaled[d] for d in QUALITY_LIST if d in scaled]
    s = [scaled[d] for d in SEMANTIC_LIST if d in scaled]
    # Divide by sum-of-weights, not count — dynamic_degree has weight 0.5
    q_wsum = sum(DIM_WEIGHT.get(d, 1) for d in QUALITY_LIST if d in scaled)
    s_wsum = sum(DIM_WEIGHT.get(d, 1) for d in SEMANTIC_LIST if d in scaled)
    qs = sum(q) / q_wsum if q else None
    ss = sum(s) / s_wsum if s else None
    total = (qs * QUALITY_WEIGHT + ss * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT) if qs and ss else None
    return {"quality_score": qs, "semantic_score": ss, "total_score": total}


def load_timing(log_dir):
    m = {}
    # Source 1: generation_log_*.json files inside the videos dir (EasyCache style)
    for p in glob.glob(os.path.join(log_dir, "generation_log_*.json")):
        with open(p) as f:
            log = json.load(f)
        for run in log.get("runs", []):
            if "time_seconds" not in run:
                continue
            mode = run["mode"]
            m.setdefault(mode, []).append(run["time_seconds"])
    # Source 2: timing_*.json files in a sibling results/ dir (MagCache style)
    results_dir = os.path.join(os.path.dirname(log_dir.rstrip("/")), "results")
    for p in glob.glob(os.path.join(results_dir, "timing_*.json")):
        with open(p) as f:
            data = json.load(f)
        mode = data.get("mode")
        if mode:
            for run in data.get("runs", []):
                if "time_seconds" in run:
                    m.setdefault(mode, []).append(run["time_seconds"])
    return {mode: {"avg_time": sum(t)/len(t), "num_videos": len(t)} for mode, t in m.items()}


def load_fidelity(metrics_dir):
    p = os.path.join(metrics_dir, "all_fidelity_results.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    # Fallback: merge per-mode files (e.g. from 2-GPU parallel run)
    r = {}
    for fpath in glob.glob(os.path.join(metrics_dir, "*_vs_*_baseline.json")):
        with open(fpath) as f:
            d = json.load(f)
        mode = d.get("mode")
        if mode:
            r[mode] = d
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", default="vbench_eval_easycache/vbench_scores")
    p.add_argument("--fidelity-dir", default="vbench_eval_easycache/fidelity_metrics")
    p.add_argument("--gen-log-dir", default="vbench_eval_easycache/videos")
    p.add_argument("--output-json", default="vbench_eval_easycache/all_comparison_results.json")
    p.add_argument("--output-csv", default="vbench_eval_easycache/all_comparison_results.csv")
    p.add_argument("--modes", default=None, help="Comma-separated list of modes (overrides ALL_MODES)")
    p.add_argument("--baseline", default=None, help="Mode name to use as baseline for speedup (default: first mode)")
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else ALL_MODES
    baseline_mode = args.baseline if args.baseline else modes[0]

    timing = load_timing(args.gen_log_dir)
    fidelity = load_fidelity(args.fidelity_dir)
    baseline_time = timing.get(baseline_mode, {}).get("avg_time")

    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print("%-30s %8s %8s %10s %8s %8s %8s" % ("Mode", "Speedup", "Latency", "VBench", "PSNR", "SSIM", "LPIPS"))
    print("-" * 80)

    rows = []
    for mode in modes:
        raw = load_vbench_scores(os.path.join(args.scores_dir, mode))
        agg = compute_aggregate(raw)
        t = timing.get(mode, {})
        fid = fidelity.get(mode, {})
        speedup = baseline_time / t["avg_time"] if baseline_time and t else None
        row = {
            "mode": MODE_LABELS.get(mode, mode),
            "speedup": "%.2fx" % speedup if speedup else "-",
            "latency": "%ds" % t["avg_time"] if t else "-",
            "vbench": "%.6f" % agg["total_score"] if agg["total_score"] else "-",
            "quality": "%.6f" % agg["quality_score"] if agg["quality_score"] else "-",
            "semantic": "%.6f" % agg["semantic_score"] if agg["semantic_score"] else "-",
            "psnr": "%.4f" % fid["psnr"]["mean"] if fid and "psnr" in fid else "-",
            "ssim": "%.4f" % fid["ssim"]["mean"] if fid and "ssim" in fid else "-",
            "lpips": "%.4f" % fid["lpips"]["mean"] if fid and "lpips" in fid else "-",
        }
        rows.append(row)
        print("%-30s %8s %8s %10s %8s %8s %8s" % (
            row["mode"], row["speedup"], row["latency"], row["vbench"],
            row["psnr"], row["ssim"], row["lpips"]))

    out_dir = os.path.dirname(args.output_json) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump({"modes": ALL_MODES, "rows": rows, "timing": timing, "fidelity": fidelity}, f, indent=2)
    print(f"\nSaved JSON to {args.output_json}")

    # Also write a flat CSV for easy import into analysis tools.
    try:
        import csv

        csv_path = args.output_csv
        fieldnames = ["mode", "speedup", "latency", "vbench", "quality", "semantic", "psnr", "ssim", "lpips"]
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved CSV to {csv_path}")
    except Exception as e:
        print(f"WARNING: Failed to write CSV ({e})")


if __name__ == "__main__":
    main()
