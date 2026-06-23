import numpy as np
import torch


class ModelWrapper:
    """Unified interface for Keras and PyTorch models."""

    def __init__(self, model, framework='keras'):
        self.model = model
        self.framework = framework

    def predict(self, x):
        if self.framework == 'keras':
            return self.model.predict(x, verbose=0)
        else:
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x) if not isinstance(x, torch.Tensor) else x
                logits = self.model(x_tensor)
                return torch.softmax(logits, dim=1).numpy()


class EnsembleDefense:
    """
    Ensemble voting defense using multiple diverse models.

    Different models make different errors on adversarial inputs.
    By combining predictions from diverse models, we get more robust predictions.
    """

    def __init__(self, models):
        """
        Args:
            models: list of ModelWrapper instances
        """
        self.models = models

    def predict(self, x):
        """
        Get ensemble prediction using soft voting.

        Each model outputs probability distribution.
        Final prediction = average of all probability distributions.
        """
        all_probs = []

        for model in self.models:
            probs = model.predict(x)
            if hasattr(probs, 'numpy'):
                probs = probs.numpy()
            all_probs.append(probs)

        # Average probabilities (soft voting)
        avg_probs = np.mean(all_probs, axis=0)

        if len(avg_probs.shape) == 1:
            avg_probs = avg_probs.reshape(1, -1)

        predictions = np.argmax(avg_probs, axis=1)
        confidence = np.max(avg_probs, axis=1)

        # Agreement score: how many models agree
        individual_preds = [np.argmax(p, axis=1) if len(p.shape) > 1 else np.array([np.argmax(p)]) for p in all_probs]
        stacked = np.stack(individual_preds, axis=0)
        agreement = np.apply_along_axis(
            lambda col: np.max(np.bincount(col.astype(int))) / len(col), 0, stacked
        )

        return {
            'predictions': predictions,
            'confidence': confidence,
            'agreement': agreement,
            'avg_probs': avg_probs
        }
