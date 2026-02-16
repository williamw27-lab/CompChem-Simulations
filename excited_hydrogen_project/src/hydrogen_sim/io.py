### saves data in /runs 

import numpy as np

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import shutil
import json

# * Create a run directory
def create_run_dir(base_dir="runs", make_latest=True):
    base = "excited_hydrogen_project" / Path(base_dir)
    base.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = base / timestamp
    run_dir.mkdir()

    if make_latest:
        latest = base / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        latest.mkdir()

    return run_dir, (base / "latest" if make_latest else None)

# * Save results (arrays)

def save_results_npz(run_dir, **arrays):
    np.savez(run_dir / "results.npz", **arrays)

# * json summary

def save_summary_json(run_dir, summary_dict):
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

# * update latest

def update_latest(run_dir, latest_dir):
    for fname in ["results.npz", "summary.json"]:
        src = run_dir / fname
        dst = latest_dir / fname
        shutil.copy2(src, dst)

# * core function

# @dataclass
# class pathSaver:
#     run_path: str
#     latest_path: str
#     summary_dict: dict
#     arrays: dict TODO: make consistent between saving arrays individually or in a single dictionary


def complete_save(summary_dict, **arrays):
    run_dir, latest_dir = create_run_dir()

    save_summary_json(run_dir=run_dir,summary_dict=summary_dict)
    save_results_npz(run_dir=run_dir,**arrays)
    update_latest(run_dir,latest_dir)

    # return pathSaver(run_path=Path(run_dir),latest_path=Path(latest_dir),summary_dict=summary_dict)