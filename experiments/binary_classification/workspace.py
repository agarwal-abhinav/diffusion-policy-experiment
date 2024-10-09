import torch
import wandb
import os
from tqdm import tqdm
from omegaconf import OmegaConf
from experiments.binary_classification.dataset import get_dataloaders
from experiments.binary_classification.classifier import MLP
from experiments.binary_classification.dataset import BinaryClassificationDataset

class BinaryClassificationWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.experiment_name = cfg.experiment.name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataloaders
        dataset = BinaryClassificationDataset(
            cfg.experiment.data_0_path, 
            cfg.experiment.data_1_path
        )
        self.train_loader, self.val_loader = get_dataloaders(
            dataset,
            cfg.experiment.batch_size,
            cfg.experiment.val_split,
        )

        # Model, optimizer, loss
        simple_model = cfg.experiment.simple_model
        self.model = MLP(cfg.experiment.input_dim, simple_model).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.experiment.lr)
        pos_weight = dataset.get_num_zeros() / dataset.get_num_ones()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))

    
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

            # Validation loss and metrics
            self.model.eval()
            epoch_val_loss = 0
            validation_accuracy = 0
            total_val_datapoints = 0
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    # loss
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels.unsqueeze(1).float())
                    epoch_val_loss += loss.item()

                    # accuracy and confusion matrix
                    predictions = (torch.sigmoid(outputs) >= 0.5).float()
                    validation_accuracy += (predictions == labels.unsqueeze(1)).sum().item()
                    total_val_datapoints += len(labels)

                    true_positives += ((predictions == 1) & (labels.unsqueeze(1) == 1)).sum().item()
                    false_positives += ((predictions == 1) & (labels.unsqueeze(1) == 0)).sum().item()
                    true_negatives += ((predictions == 0) & (labels.unsqueeze(1) == 0)).sum().item()
                    false_negatives += ((predictions == 0) & (labels.unsqueeze(1) == 1)).sum().item()

            epoch_train_loss /= len(self.train_loader)
            epoch_val_loss /= len(self.val_loader)
            validation_accuracy /= total_val_datapoints

            # Calculate percentages for true/false positives/negatives
            predicted_positives = true_positives + false_positives
            predicted_negatives = true_negatives + false_negatives

            true_positive_percentage = true_positives / predicted_positives if predicted_positives > 0 else 0
            false_positive_percentage = false_positives / predicted_positives if predicted_positives > 0 else 0
            true_negative_percentage = true_negatives / predicted_negatives if predicted_negatives > 0 else 0
            false_negative_percentage = false_negatives / predicted_negatives if predicted_negatives > 0 else 0

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
                    "val_accuracy": validation_accuracy,
                    "true_positive_percentage": true_positive_percentage,
                    # "false_positive_percentage": false_positive_percentage,
                    "true_negative_percentage": true_negative_percentage,
                    # "false_negative_percentage": false_negative_percentage
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
            predicted_labels = (torch.sigmoid(outputs) >= 0.5)
            return outputs.cpu().numpy(), predicted_labels.cpu().numpy()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)