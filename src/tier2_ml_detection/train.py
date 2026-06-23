import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.tier2_ml_detection.models import build_dnn, build_cnn, build_lstm, get_training_callbacks
from src.tier2_ml_detection.feature_extractor import FeatureExtractor
from src.utils.config import resolve_path


class Tier2Trainer:
    """Train and compare DNN, CNN, LSTM models for Tier 2 detection."""

    def __init__(self, config):
        self.config = config
        self.training_config = config['tier2']['training']

    def train_model(self, model_type, X_train, y_train, X_val, y_val, n_classes):
        """Train a single model type."""
        n_features = X_train.shape[1]
        extractor = FeatureExtractor(model_type, n_features)

        # Reshape data
        X_train_r = extractor.reshape(X_train)
        X_val_r = extractor.reshape(X_val)

        # Build model
        input_shape = extractor.get_input_shape()
        dropout = self.training_config['dropout_rate']

        if model_type.upper() == 'DNN':
            model = build_dnn(input_shape, n_classes, dropout)
        elif model_type.upper() == 'CNN':
            model = build_cnn(input_shape, n_classes, dropout)
        elif model_type.upper() == 'LSTM':
            model = build_lstm(input_shape, n_classes, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Model save path
        model_dir = resolve_path('models/tier2')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{model_type.lower()}_model.h5')

        # Train
        callbacks = get_training_callbacks(model_path)
        history = model.fit(
            X_train_r, y_train,
            validation_data=(X_val_r, y_val),
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        val_loss, val_acc = model.evaluate(X_val_r, y_val, verbose=0)

        return {
            'model': model,
            'model_type': model_type,
            'history': history.history,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'model_path': model_path
        }

    def train_all_models(self, data_dict):
        """Train DNN, CNN, LSTM and compare."""
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        n_classes = data_dict['n_classes']

        results = {}
        for model_type in ['DNN', 'CNN', 'LSTM']:
            print(f"\n{'='*50}")
            print(f"Training {model_type} model...")
            print(f"{'='*50}")

            result = self.train_model(
                model_type, X_train, y_train, X_val, y_val, n_classes
            )
            results[model_type] = result
            print(f"{model_type} - Val Accuracy: {result['val_accuracy']:.4f}")

        # Find and save best model
        best_type = max(results, key=lambda k: results[k]['val_accuracy'])
        best_model = results[best_type]['model']

        best_path = resolve_path(self.config['tier2']['model_path'])
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        best_model.save(best_path)

        print(f"\nBest model: {best_type} (Val Accuracy: {results[best_type]['val_accuracy']:.4f})")
        print(f"Saved to: {best_path}")

        return results, best_type
