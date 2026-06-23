import numpy as np
import time
from src.tier3_adversarial_defense.input_transformation import feature_squeezing
from src.tier3_adversarial_defense.ensemble_defense import EnsembleDefense, ModelWrapper


class Tier3AdversarialDefense:
    """
    Main Tier 3 module: Adversarial robustness verification.

    Performs:
    1. Adversarial detection via feature squeezing
    2. Robust classification via ensemble voting
    3. Confidence scoring
    """

    def __init__(self, robust_model, ensemble_models, detection_threshold=0.1):
        self.robust_model = robust_model
        self.ensemble = EnsembleDefense(ensemble_models)
        self.threshold = detection_threshold

    def detect_adversarial(self, x):
        """
        Detect if input is adversarial using feature squeezing.

        Compare model predictions on original vs squeezed input.
        Large difference indicates adversarial perturbation.
        """
        x_input = x.reshape(1, -1) if len(x.shape) == 1 else x

        # Original prediction
        pred_original = self.robust_model.predict(x_input)
        if hasattr(pred_original, 'numpy'):
            pred_original = pred_original.numpy()
        pred_original = np.array(pred_original).flatten()

        # Squeezed prediction
        x_squeezed = feature_squeezing(x)
        x_squeezed_input = x_squeezed.reshape(1, -1) if len(x_squeezed.shape) == 1 else x_squeezed
        pred_squeezed = self.robust_model.predict(x_squeezed_input)
        if hasattr(pred_squeezed, 'numpy'):
            pred_squeezed = pred_squeezed.numpy()
        pred_squeezed = np.array(pred_squeezed).flatten()

        # L1 distance between prediction distributions
        distance = np.abs(pred_original - pred_squeezed).sum()

        is_adversarial = distance > self.threshold
        return is_adversarial, float(distance)

    def detect_and_classify(self, x):
        """
        Full Tier 3 pipeline: detect adversarial + classify robustly.

        Returns:
            Detection result dict
        """
        start_time = time.time()

        # Step 1: Detect adversarial
        is_adv, adv_score = self.detect_adversarial(x)

        # Step 2: Robust classification via ensemble
        x_input = x.reshape(1, -1) if len(x.shape) == 1 else x
        ensemble_result = self.ensemble.predict(x_input)

        prediction = int(ensemble_result['predictions'][0])
        confidence = float(ensemble_result['confidence'][0])
        agreement = float(ensemble_result['agreement'][0])

        # Step 3: Determine severity
        if is_adv:
            severity = 'CRITICAL'
        elif confidence > 0.9 and agreement > 0.8:
            severity = 'HIGH'
        elif confidence > 0.7:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        detection_time = (time.time() - start_time) * 1000

        return {
            'is_adversarial': is_adv,
            'adversarial_score': adv_score,
            'prediction': prediction,
            'confidence': confidence,
            'agreement': agreement,
            'severity': severity,
            'tier': 3,
            'detection_time_ms': detection_time
        }
