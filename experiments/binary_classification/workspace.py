import torch
import wandb
import os
from tqdm import tqdm
from omegaconf import OmegaConf
from experiments.binary_classification.dataset import get_dataloaders
from experiments.binary_classification.classifier import MLP

class BinaryClassificationWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.experiment_name = cfg.experiment.name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataloaders
        self.train_loader, self.val_loader = get_dataloaders(
            cfg.experiment.data_0_path,
            cfg.experiment.data_1_path,
            cfg.experiment.batch_size,
            cfg.experiment.val_split,
        )

        # Model, optimizer, loss
        self.model = MLP(cfg.experiment.input_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.experiment.lr)
        self.loss_fn = torch.nn.BCELoss()

    
    def run(self):
        wandb.init(project=self.experiment_name, config=self.cfg)

        best_val_loss = float("inf")
        best_val_loss_epoch = -1
        for epoch in tqdm(range(self.cfg.experiment.num_epochs), desc="Epochs"):
            self.model.train()

            # Training
            epoch_train_loss = 0
            for inputs, labels in self.train_loader:
                # Forward pass
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels.unsqueeze(1).float())
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            # Validation loss
            self.model.eval()
            epoch_val_loss = 0
            validation_accuracy = 0
            total_val_datapoints = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    # loss
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels.unsqueeze(1).float())
                    epoch_val_loss += loss.item()

                    # accuracy
                    predictions = (outputs >= 0.5).float()
                    validation_accuracy += (predictions == labels.unsqueeze(1)).sum().item()
                    total_val_datapoints += len(labels)

            epoch_train_loss /= len(self.train_loader)
            epoch_val_loss /= len(self.val_loader)
            validation_accuracy /= total_val_datapoints

            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_loss_epoch = epoch
                # self.save_model("best_val_loss.pth")

            # Log metrics
            wandb.log(
                {
                    "train_loss": epoch_train_loss, 
                    "val_loss": epoch_val_loss,
                    "val_accuracy": validation_accuracy
                },
            )
        
        # Save models
        # self.save_model("final_model.pth")
        # os.rename("best_val_loss.pth", f"best_val_loss_epoch={best_val_loss_epoch}.pth")
        wandb.finish()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, inputs):
        self.model.eval()
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            predicted_labels = (outputs >= 0.5)
            return outputs.cpu().numpy(), predicted_labels.cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
