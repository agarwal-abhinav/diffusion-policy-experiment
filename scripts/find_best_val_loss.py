import os
import json
import argparse

def find_best_val_loss(wandb_run_dir):
    """
    Find the lowest val_loss_0 from logs.json.txt in a given wandb run directory.

    Args:
        wandb_run_dir (str): Path to the wandb run directory.

    Prints:
        - The minimum val_loss_0 value
        - The number of val_loss_0 measurements taken
        - The epoch where the minimum occurred
        - The total number of unique epochs recorded
    """
    log_file = os.path.join(wandb_run_dir, "logs.json.txt")
    
    if not os.path.exists(log_file):
        print(f"Error: logs.json.txt not found in {wandb_run_dir}")
        return
    
    min_val_loss = float("inf")
    best_epoch = None
    val_loss_count = 0
    epochs_seen = set()

    with open(log_file, "r") as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                if "epoch" in log_entry:
                    epochs_seen.add(log_entry["epoch"])
                if "val_loss_0" in log_entry:
                    val_loss_0 = log_entry["val_loss_0"]
                    val_loss_count += 1
                    if val_loss_0 < min_val_loss:
                        min_val_loss = val_loss_0
                        best_epoch = log_entry.get("epoch", "Unknown")
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()}")

    total_epochs = len(epochs_seen)

    if val_loss_count > 0:
        print(f"Total val_loss_0 measurements: {val_loss_count}")
        print(f"Lowest val_loss_0: {min_val_loss:.6f} at epoch {best_epoch}")
        print(f"Total number of unique epochs: {total_epochs}")
    else:
        print("No val_loss_0 measurements found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find lowest val_loss_0 in wandb logs.")
    parser.add_argument("wandb_run_dir", type=str, help="Path to the wandb run directory")
    args = parser.parse_args()

    find_best_val_loss(args.wandb_run_dir)
