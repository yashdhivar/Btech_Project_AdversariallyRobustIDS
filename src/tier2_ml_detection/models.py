import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten,
    LSTM, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def build_dnn(input_dim, num_classes, dropout_rate=0.3):
    """
    Deep Neural Network for tabular traffic data.

    Architecture:
    - Input -> Dense(256) -> BN -> Dropout
    - Dense(128) -> BN -> Dropout
    - Dense(64) -> BN -> Dropout
    - Dense(32) -> Output
    """
    output_activation = 'softmax' if num_classes > 2 else 'sigmoid'
    output_units = num_classes if num_classes > 2 else 1

    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation='relu'),
        Dense(output_units, activation=output_activation)
    ])

    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


def build_cnn(input_shape, num_classes, dropout_rate=0.3):
    """
    1D Convolutional Neural Network for pattern recognition.

    Requires input reshaped to (n_features, 1).
    """
    output_activation = 'softmax' if num_classes > 2 else 'sigmoid'
    output_units = num_classes if num_classes > 2 else 1

    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(output_units, activation=output_activation)
    ])

    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


def build_lstm(input_shape, num_classes, dropout_rate=0.3):
    """
    LSTM Network for sequential pattern detection.

    Requires input reshaped to (timesteps, features).
    """
    output_activation = 'softmax' if num_classes > 2 else 'sigmoid'
    output_units = num_classes if num_classes > 2 else 1

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),

        LSTM(64, return_sequences=False),
        Dropout(dropout_rate),

        Dense(32, activation='relu'),
        Dense(output_units, activation=output_activation)
    ])

    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


def get_training_callbacks(model_path):
    """Standard callbacks for all models."""
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    ]
