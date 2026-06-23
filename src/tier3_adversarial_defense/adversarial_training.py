import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.adversarial_attacks.fgsm import fgsm_attack
from src.adversarial_attacks.pgd import pgd_attack
from src.utils.config import resolve_path


class AdversarialTrainer:
    """
    Trains models to be robust against adversarial attacks.

    During each training batch, generates adversarial examples
    on-the-fly and mixes them with clean samples.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['tier2']['training']['learning_rate']
        )

    def train_epoch(self, dataloader):
        """Train one epoch with adversarial samples."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Generate adversarial examples
            self.model.eval()
            x_fgsm = fgsm_attack(
                self.model, x_batch, y_batch,
                epsilon=self.config['adversarial_attacks']['fgsm']['epsilon']
            )
            x_pgd = pgd_attack(
                self.model, x_batch, y_batch,
                epsilon=self.config['adversarial_attacks']['pgd']['epsilon'],
                alpha=self.config['adversarial_attacks']['pgd']['alpha'],
                num_iterations=10
            )
            self.model.train()

            # Mix: 40% clean + 30% FGSM + 30% PGD
            batch_size = x_batch.size(0)
            clean_size = int(0.4 * batch_size)
            fgsm_size = int(0.3 * batch_size)

            x_mixed = torch.cat([
                x_batch[:clean_size],
                x_fgsm[clean_size:clean_size + fgsm_size],
                x_pgd[clean_size + fgsm_size:]
            ])
            y_mixed = torch.cat([
                y_batch[:clean_size],
                y_batch[clean_size:clean_size + fgsm_size],
                y_batch[clean_size + fgsm_size:]
            ])

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x_mixed)
            loss = F.cross_entropy(output, y_mixed)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == y_mixed).sum().item()
            total += y_mixed.size(0)

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        return avg_loss, accuracy

    def evaluate(self, dataloader):
        """Evaluate on clean data."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item()
                correct += (output.argmax(1) == y).sum().item()
                total += y.size(0)

        return total_loss / max(len(dataloader), 1), correct / max(total, 1)

    def train(self, train_loader, val_loader, epochs=None):
        """Full adversarial training loop."""
        epochs = epochs or self.config['adversarial_training']['epochs']
        best_val_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = resolve_path(self.config['tier3']['robust_model_path'])
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)

        return best_val_acc
