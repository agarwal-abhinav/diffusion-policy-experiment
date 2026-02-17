import os
import wandb
import pandas as pd

# --- CONFIGURATION ---
# Replace these with your details
ENTITY = "rlg_abhinav" 
PROJECT = "long_context_pushing_iros_two_modes_skip"
CHECKPOINT_DIR = "data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_24/recent_plus_72_mode_4_0/47_obs/"
# ---------------------

def get_run_id_from_dir(base_dir):
    """
    Scans the local wandb folder to find the Run ID.
    Expects structure: base_dir/wandb/run-DATE_TIME-RUNID
    """
    wandb_dir = os.path.join(base_dir, "wandb")
    
    if not os.path.exists(wandb_dir):
        raise FileNotFoundError(f"Could not find a 'wandb' folder inside {base_dir}")

    # List all directories in wandb/
    subdirs = [d for d in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, d))]
    
    # Filter for folders starting with "run-"
    run_folders = [d for d in subdirs if d.startswith("run-")]
    
    if not run_folders:
        raise ValueError("No 'run-*' folders found in the wandb directory.")

    # Extract ID from the first folder found.
    # Format is usually: run-YYYYMMDD_HHMMSS-RUNID
    # We split by '-' and take the last part.
    # Since you confirmed the ID is the same for all fragments, we just need one.
    example_folder = run_folders[0]
    run_id = example_folder.split("-")[-1]
    
    return run_id

def analyze_best_epochs(entity, project, run_id):
    print(f"Fetching history for Run ID: {run_id}...")
    
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except wandb.errors.CommError:
        print("Error: Could not find run. Check your Entity/Project names.")
        return

    # Fetch specific columns. 
    # We use a large 'samples' number to ensure we get all steps if the run is long.
    # wandb usually samples history, but for exact epoch finding, we want the full set.
    history = run.history(keys=["epoch", "val_loss_0", "val_ddim_mse_0"], samples=100000)
    
    # Drop rows where metrics might be NaN (e.g. if logged at different steps)
    df = pd.DataFrame(history)
    
    # Ensure we actually have data
    if "val_loss_0" not in df.columns:
        print("Error: 'val_loss' not found in run history.")
        return

    # 1. Best val_loss
    print("\n--- Top 5 Epochs by Lowest Validation Loss ---")
    best_loss = df.sort_values("val_loss_0", ascending=True).dropna(subset=["val_loss_0"]).head(5)
    print(best_loss[["epoch", "val_loss_0"]].to_string(index=False))

    # 2. Best val_ddim_mse (Check if it exists first)
    if "val_ddim_mse_0" in df.columns:
        print("\n--- Top 5 Epochs by Lowest val_ddim_mse ---")
        best_mse = df.sort_values("val_ddim_mse_0", ascending=True).dropna(subset=["val_ddim_mse_0"]).head(5)
        print(best_mse[["epoch", "val_ddim_mse_0"]].to_string(index=False))
    else:
        print("\nWarning: 'val_ddim_mse' not found in logs.")

# --- EXECUTION ---
try:
    # 1. Get the ID from the local folder
    detected_run_id = get_run_id_from_dir(CHECKPOINT_DIR)
    print(f"Detected Run ID from local files: {detected_run_id}")
    
    # 2. Fetch and Analyze
    analyze_best_epochs(ENTITY, PROJECT, detected_run_id)

except Exception as e:
    print(f"An error occurred: {e}")