"""
Given a training output directory, find the wandb run and report
train_loss, val_loss, and val_ddim_mse at the epochs/steps
corresponding to saved checkpoints.

Usage:
    python checkpoint_metrics.py \
        --project long_context_pushing_iros_two_modes \
        --dir data/outputs/iros/long_context_planar_pushing/...

    # Override entity if needed (default: rlg_abhinav)
    python checkpoint_metrics.py \
        --entity someone_else \
        --project my_project \
        --dir /path/to/run
"""

import os
import re
import argparse
import wandb
import pandas as pd


def get_run_id_from_dir(base_dir):
    """Extract wandb Run ID from local wandb/ folder."""
    wandb_dir = os.path.join(base_dir, "wandb")
    if not os.path.exists(wandb_dir):
        raise FileNotFoundError(f"No 'wandb' folder in {base_dir}")

    run_folders = [
        d for d in os.listdir(wandb_dir)
        if os.path.isdir(os.path.join(wandb_dir, d)) and d.startswith("run-")
    ]
    if not run_folders:
        raise ValueError("No 'run-*' folders found in wandb directory.")

    return run_folders[0].split("-")[-1]


def get_checkpoint_identifiers(base_dir):
    """
    Parse checkpoint filenames and return list of (display_name, short_name, epoch, step).
    epoch or step may be None if not parseable.
    short_name is e.g. "epoch=0152" or "step=85000" or "latest".
    """
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"No 'checkpoints' folder in {base_dir}")

    epoch_re = re.compile(r"epoch=(\d+)")
    step_re = re.compile(r"step=(\d+)")

    checkpoints = []
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".ckpt"):
            continue

        epoch = None
        step = None
        short_name = fname.replace(".ckpt", "")

        m = epoch_re.search(fname)
        if m:
            epoch = int(m.group(1))
            short_name = f"epoch={epoch}"

        m = step_re.search(fname)
        if m:
            step = int(m.group(1))
            if epoch is None:
                short_name = f"step={step}"

        if fname == "latest.ckpt":
            short_name = "latest"

        checkpoints.append((fname, short_name, epoch, step))

    return checkpoints


def derive_csv_name(dir_path):
    """
    Derive a CSV filename from the directory path.
    If path contains data_XX and XX_obs, capture those.
    Otherwise use a generic name.
    """
    dir_path = dir_path.rstrip("/")

    data_match = re.search(r"data_(\d+)", dir_path)
    obs_match = re.search(r"(\d+)_obs", dir_path)

    # Also try to capture variable context config names
    basename = os.path.basename(dir_path)

    if data_match and obs_match:
        return f"metrics_data_{data_match.group(1)}_{obs_match.group(1)}_obs.csv"
    elif data_match:
        return f"metrics_data_{data_match.group(1)}_{basename}.csv"
    else:
        return f"metrics_{basename}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Report wandb metrics at saved checkpoint epochs/steps.")
    parser.add_argument("--entity", default="rlg_abhinav", help="wandb entity (default: rlg_abhinav)")
    parser.add_argument("--project", required=True, help="wandb project")
    parser.add_argument("--dir", required=True, help="Training output directory")
    parser.add_argument("--csv", default=None, help="Output CSV path (auto-derived if not given)")
    args = parser.parse_args()

    # 1. Find run ID
    run_id = get_run_id_from_dir(args.dir)
    print(f"Run ID: {run_id}")

    # 2. Pull wandb history
    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{run_id}")
    history = run.history(
        keys=["epoch", "global_step", "train_loss", "val_loss", "val_ddim_mse"],
        samples=500000,
    )
    df = pd.DataFrame(history)

    # 3. Parse checkpoints
    checkpoints = get_checkpoint_identifiers(args.dir)
    if not checkpoints:
        print("No checkpoints found.")
        return

    # 4. Match each checkpoint to wandb metrics
    rows = []
    print(f"\n{'Checkpoint':<65} {'Epoch':>6} {'Step':>8} {'train_loss':>12} {'val_loss':>12} {'val_ddim_mse':>14}")
    print("-" * 125)

    for fname, short_name, epoch, step in checkpoints:
        row = None

        if epoch is not None and "epoch" in df.columns:
            matches = df[df["epoch"] == epoch].dropna(subset=["val_loss"], how="all")
            if len(matches) > 0:
                row = matches.iloc[-1]

        if row is None and step is not None and "global_step" in df.columns:
            diffs = (df["global_step"] - step).abs()
            closest_idx = diffs.idxmin()
            if diffs[closest_idx] <= 500:
                row = df.iloc[closest_idx]

        if row is not None:
            t_loss = row.get('train_loss', float('nan'))
            v_loss = row.get('val_loss', float('nan'))
            v_ddim = row.get('val_ddim_mse', float('nan'))
            e = int(row["epoch"]) if pd.notna(row.get("epoch")) else None
            s = int(row["global_step"]) if pd.notna(row.get("global_step")) else None

            t_str = f"{t_loss:.6f}" if pd.notna(t_loss) else "N/A"
            v_str = f"{v_loss:.6f}" if pd.notna(v_loss) else "N/A"
            d_str = f"{v_ddim:.6f}" if pd.notna(v_ddim) else "N/A"
            e_str = str(e) if e is not None else "?"
            s_str = str(s) if s is not None else "?"

            print(f"{fname:<65} {e_str:>6} {s_str:>8} {t_str:>12} {v_str:>12} {d_str:>14}")
            rows.append({
                "checkpoint": short_name,
                "epoch": e,
                "step": s,
                "train_loss": t_loss if pd.notna(t_loss) else None,
                "val_loss": v_loss if pd.notna(v_loss) else None,
                "val_ddim_mse": v_ddim if pd.notna(v_ddim) else None,
            })
        else:
            e_str = str(epoch) if epoch is not None else "?"
            s_str = str(step) if step is not None else "?"
            print(f"{fname:<65} {e_str:>6} {s_str:>8} {'N/A':>12} {'N/A':>12} {'N/A':>14}")
            rows.append({
                "checkpoint": short_name,
                "epoch": epoch,
                "step": step,
                "train_loss": None,
                "val_loss": None,
                "val_ddim_mse": None,
            })

    # 5. Print overall best
    print(f"\n{'='*125}")
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        best = df.loc[df["val_loss"].idxmin()]
        print(f"Best val_loss:     {best['val_loss']:.6f} at epoch {int(best['epoch'])}, step {int(best['global_step'])}")
    if "val_ddim_mse" in df.columns and df["val_ddim_mse"].notna().any():
        best = df.loc[df["val_ddim_mse"].idxmin()]
        print(f"Best val_ddim_mse: {best['val_ddim_mse']:.6f} at epoch {int(best['epoch'])}, step {int(best['global_step'])}")

    # 6. Save CSV
    csv_path = args.csv if args.csv else derive_csv_name(args.dir)
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
