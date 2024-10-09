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