import numpy as np
import tensorflow as tf
import time

from src.tier2_ml_detection.feature_extractor import FeatureExtractor


class Tier2MLDetector:
    """
    ML-based anomaly detection module.
    Uses trained DNN/CNN/LSTM to classify network traffic.
    """

    CLASS_MAP = {
        0: 'BENIGN', 1: 'DoS', 2: 'Probe',
        3: 'R2L', 4: 'U2R'
    }

    def __init__(self, model_path, model_type='DNN', confidence_threshold=0.5):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = confidence_threshold
        self.model_type = model_type
        # Infer n_features from model input shape
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        n_features = input_shape[-1] if len(input_shape) > 2 else input_shape[1]
        self.extractor = FeatureExtractor(model_type, n_features)

    def detect(self, features: np.ndarray) -> dict:
        """
        Classify a single preprocessed traffic sample.

        Args:
            features: Preprocessed feature vector (numpy array)

        Returns:
            Detection result dict
        """
        start_time = time.time()

        # Reshape
        input_data = self.extractor.reshape(features)

        # Predict
        probabilities = self.model.predict(input_data, verbose=0)[0]

        # Handle binary vs multi-class output
        if len(probabilities.shape) == 0 or probabilities.shape == ():
            # Binary sigmoid output
            confidence = float(probabilities)
            is_attack = confidence >= self.threshold
            predicted_class = 1 if is_attack else 0
            probabilities_list = [1 - confidence, confidence]
        else:
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            is_attack = predicted_class != 0 and confidence >= self.threshold
            probabilities_list = probabilities.tolist()

        detection_time = (time.time() - start_time) * 1000

        return {
            'is_attack': is_attack,
            'predicted_class': predicted_class,
            'attack_type': self._class_to_name(predicted_class),
            'confidence': confidence,
            'probabilities': probabilities_list,
            'tier': 2,
            'detection_time_ms': detection_time
        }

    def detect_batch(self, features_batch: np.ndarray) -> list:
        """Classify a batch of traffic samples."""
        input_data = self.extractor.reshape(features_batch)
        probabilities = self.model.predict(input_data, verbose=0)
        results = []

        for prob in probabilities:
            if prob.shape == () or len(prob.shape) == 0:
                confidence = float(prob)
                predicted_class = 1 if confidence >= self.threshold else 0
            else:
                predicted_class = int(np.argmax(prob))
                confidence = float(prob[predicted_class])

            is_attack = predicted_class != 0 and confidence >= self.threshold
            results.append({
                'is_attack': is_attack,
                'predicted_class': predicted_class,
                'attack_type': self._class_to_name(predicted_class),
                'confidence': confidence,
                'tier': 2,
            })

        return results

    def _class_to_name(self, class_id):
        """Map class ID to attack name."""
        return self.CLASS_MAP.get(class_id, f'Unknown-{class_id}')
