#!/usr/bin/env python3
"""
Remove stale log entries for the 3 corrupted cached modes.
Keeps baseline entries intact (correct timing).
Run once before re-generating cached mode videos.
"""
import json
import os
import glob

STALE_MODES = {"easycache_fixed_0.025", "easycache_fixed_0.050", "easycache_adaptive"}
log_dir = os.path.join(os.path.dirname(__file__), "videos")

logs = sorted(glob.glob(os.path.join(log_dir, "generation_log_*.json")))
if not logs:
    print("No logs found.")
else:
    for log_path in logs:
        with open(log_path) as f:
            data = json.load(f)
        before = len(data["runs"])
        data["runs"] = [r for r in data["runs"] if r["mode"] not in STALE_MODES]
        data["completed_keys"] = [
            k for k in data["completed_keys"]
            if not any(k.startswith(m + "|") for m in STALE_MODES)
        ]
        after = len(data["runs"])
        tmp = log_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, log_path)
        print(f"{os.path.basename(log_path)}: removed {before - after} stale entries, kept {after}")
