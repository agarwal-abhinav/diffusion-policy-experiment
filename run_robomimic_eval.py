#!/usr/bin/env python3
import argparse
import os
import re
import sys
import subprocess
from pathlib import Path

EPOCH_RE = re.compile(r"epoch=(\d+)")

def infer_low_level(full_path: Path, high_level: Path) -> str:
    try:
        low = str(full_path.resolve().relative_to(high_level.resolve()))
        return low
    except Exception:
        # Fallback: use the tail directory name if the provided high_level
        # is not a prefix of path.
        return full_path.name

def checkpoint_name(ckpt_path: Path) -> str:
    name = ckpt_path.name
    if name == "latest.ckpt":
        return "latest"
    m = EPOCH_RE.search(name)
    if m:
        # Return integer epoch (strip leading zeros)
        return str(int(m.group(1)))
    # Fallback: strip extension
    return ckpt_path.stem

def sort_key(ckpt_path: Path):
    """Order by epoch if present; put 'latest.ckpt' last; otherwise by name."""
    if ckpt_path.name == "latest.ckpt":
        return (float("inf"), ckpt_path.name)
    m = EPOCH_RE.search(ckpt_path.name)
    if m:
        return (int(m.group(1)), ckpt_path.name)
    return (10**12, ckpt_path.name)  # unknown epoch after numbered, before 'latest'

def main():
    p = argparse.ArgumentParser(description="Run eval.py sequentially over all checkpoints.")
    p.add_argument("--path", required=True, type=Path,
                   help="Full path to HIGH_LEVEL/LOW_LEVEL (e.g., data/outputs/robomimic/tool_hang/basic_training/2_obs)")
    p.add_argument("--high-level", type=Path, default=Path("data/outputs/robomimic"),
                   help="High-level root used to compute LOW_LEVEL (default: data/outputs/robomimic)")
    p.add_argument("--eval-script", default="eval.py",
                   help="Path to eval.py (default: eval.py in current working dir)")
    p.add_argument("--output-root", type=Path, default=Path("data/eval_output/robomimic_eval"),
                   help="Root of eval outputs (default: data/eval_output/robomimic_eval)")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    p.add_argument("--skip-existing", action="store_true", default=False,
                   help="Skip checkpoints whose output_dir already exists and is non-empty")
    args = p.parse_args()

    base = args.path
    ckpt_dir = base / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"Error: checkpoints directory not found: {ckpt_dir}", file=sys.stderr)
        sys.exit(2)

    low_level = infer_low_level(base, args.high_level)
    out_base = args.output_root / low_level

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=sort_key)
    if not ckpts:
        print(f"No .ckpt files found under {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ckpts)} checkpoints in {ckpt_dir}")
    print(f"LOW_LEVEL = {low_level}")
    print(f"Output root = {out_base}")

    for i, ckpt in enumerate(ckpts, 1):
        name = checkpoint_name(ckpt)
        outdir = out_base / name
        # outdir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and any(outdir.iterdir()):
            print(f"[{i}/{len(ckpts)}] Skipping {ckpt.name} -> {outdir} (already has contents)")
            continue

        cmd = [
            sys.executable,  # current python
            args.eval_script,
            "--checkpoint", str(ckpt),
            "--output_dir", str(outdir),
        ]

        print(f"[{i}/{len(ckpts)}] Running: {' '.join(cmd)}")
        if args.dry_run:
            continue

        # Run sequentially; raise if eval fails
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}", file=sys.stderr)
            sys.exit(e.returncode)

    print("All evals completed successfully.")

if __name__ == "__main__":
    main()
