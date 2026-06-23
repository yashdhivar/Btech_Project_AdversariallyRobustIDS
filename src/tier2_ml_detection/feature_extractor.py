import numpy as np


class FeatureExtractor:
    """Reshape features for different model architectures."""

    def __init__(self, model_type: str, n_features: int):
        self.model_type = model_type.upper()
        self.n_features = n_features

    def reshape(self, X: np.ndarray) -> np.ndarray:
        """Reshape input for the target model type."""
        if self.model_type == 'DNN':
            return self._reshape_for_dnn(X)
        elif self.model_type == 'CNN':
            return self._reshape_for_cnn(X)
        elif self.model_type == 'LSTM':
            return self._reshape_for_lstm(X)
        else:
            return X

    def _reshape_for_dnn(self, X):
        """DNN expects (samples, features)."""
        if len(X.shape) == 1:
            return X.reshape(1, -1)
        return X

    def _reshape_for_cnn(self, X):
        """CNN expects (samples, features, 1)."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return X.reshape(X.shape[0], X.shape[1], 1)

    def _reshape_for_lstm(self, X):
        """LSTM expects (samples, 1, features) for single-timestep input."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return X.reshape(X.shape[0], 1, X.shape[1])

    def get_input_shape(self):
        """Return the input shape for model construction."""
        if self.model_type == 'DNN':
            return self.n_features
        elif self.model_type == 'CNN':
            return (self.n_features, 1)
        elif self.model_type == 'LSTM':
            return (1, self.n_features)
        return self.n_features
