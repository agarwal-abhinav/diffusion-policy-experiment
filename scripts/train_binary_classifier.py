"""
python scripts/train_binary_classifier.py --config-name <config name> \
    experiments.data_0_path=<path> experiments.data_1_path=<path>
"""
import hydra
from experiments.binary_classification.workspace import BinaryClassificationWorkspace

@hydra.main(
    config_path="../experiments/binary_classification/config", 
)
def main(cfg):
    workspace = BinaryClassificationWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()