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
    qs = sum(q) / len(q) if q else None
    ss = sum(s) / len(s) if s else None
    total = (qs * QUALITY_WEIGHT + ss * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT) if qs and ss else None
    return {"quality_score": qs, "semantic_score": ss, "total_score": total}


def load_timing(log_dir):
    m = {}
    for p in glob.glob(os.path.join(log_dir, "generation_log_*.json")):
        with open(p) as f:
            log = json.load(f)
        for run in log.get("runs", []):
            if "time_seconds" not in run:
                continue
            mode = run["mode"]
            m.setdefault(mode, []).append(run["time_seconds"])
    return {mode: {"avg_time": sum(t)/len(t), "num_videos": len(t)} for mode, t in m.items()}


def load_fidelity(metrics_dir):
    p = os.path.join(metrics_dir, "all_fidelity_results.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    # Fallback: merge per-mode files (e.g. from 2-GPU parallel run)
    r = {}
    for fpath in glob.glob(os.path.join(metrics_dir, "*_vs_easycache_baseline.json")):
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
    args = p.parse_args()

    timing = load_timing(args.gen_log_dir)
    fidelity = load_fidelity(args.fidelity_dir)
    baseline_time = timing.get("easycache_baseline", {}).get("avg_time")

    print("=" * 80)
    print("EasyCache Evaluation Results")
    print("=" * 80)

    rows = []
    for mode in ALL_MODES:
        raw = load_vbench_scores(os.path.join(args.scores_dir, mode))
        agg = compute_aggregate(raw)
        t = timing.get(mode, {})
        fid = fidelity.get(mode, {})
        speedup = baseline_time / t["avg_time"] if baseline_time and t else None
        row = {
            "mode": MODE_LABELS.get(mode, mode),
            "speedup": f"{speedup:.2f}x" if speedup else "—",
            "latency": f"{t['avg_time']:.0f}s" if t else "—",
            "vbench": f"{agg['total_score']*100:.2f}%" if agg["total_score"] else "—",
            "psnr": f"{fid['psnr']['mean']:.4f}" if fid and "psnr" in fid else "—",
            "ssim": f"{fid['ssim']['mean']:.4f}" if fid and "ssim" in fid else "—",
            "lpips": f"{fid['lpips']['mean']:.4f}" if fid and "lpips" in fid else "—",
        }
        rows.append(row)
        print(f"{row['mode']:<22} {row['speedup']:>8} {row['latency']:>8} {row['vbench']:>10} {row['psnr']:>8} {row['ssim']:>8} {row['lpips']:>8}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"modes": ALL_MODES, "rows": rows, "timing": timing, "fidelity": fidelity}, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
