import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
import joblib

from src.utils.config import resolve_path
from src.preprocessing.data_loader import CATEGORY_MAP


class DataPreprocessor:
    """
    Complete data preprocessing pipeline for IDS datasets.
    Handles loading, cleaning, encoding, scaling, imbalance, and splitting.
    """

    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = None
        self.feature_names = None
        self.n_features = None

    def clean_data(self, df):
        """Handle missing values, infinities, and duplicates."""
        df = df.copy()

        # Replace infinity with NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Fill missing values
        method = self.config['preprocessing']['handle_missing']
        if method == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            df.dropna(inplace=True)

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        return df

    def encode_labels(self, df):
        """
        Encode labels:
        - Binary: Normal=0, Attack=1
        - Multi-class: each category gets a unique integer
        """
        df = df.copy()

        # Binary encoding
        df['label_binary'] = (df['attack_category'] != 'Normal').astype(int)

        # Multi-class encoding using CATEGORY_MAP
        df['label_multiclass'] = df['attack_category'].map(CATEGORY_MAP)

        # Handle unknown categories
        df['label_multiclass'] = df['label_multiclass'].fillna(0).astype(int)

        return df

    def encode_categorical(self, df):
        """One-hot encode categorical features."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Remove label columns from encoding
        exclude = ['label', 'attack_category', 'split']
        categorical_cols = [c for c in categorical_cols if c not in exclude]

        if self.config['preprocessing']['encoding'] == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        else:
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        return df

    def scale_features(self, X_train, X_val, X_test):
        """Standardize or min-max scale numerical features."""
        scaling = self.config['preprocessing']['scaling']

        if scaling == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def select_features(self, X_train, y_train, feature_names, k=35):
        """Select top-k features using mutual information."""
        k = min(k, X_train.shape[1])  # Can't select more features than available

        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif, k=k
        )
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        mask = self.feature_selector.get_support()
        self.selected_features = [f for f, m in zip(feature_names, mask) if m]
        self.n_features = k
        return X_train_selected

    def handle_imbalance(self, X_train, y_train):
        """Apply SMOTE to balance classes."""
        if self.config['preprocessing']['imbalance_method'] == 'smote':
            class_counts = np.bincount(y_train.astype(int))
            min_samples = class_counts[class_counts > 0].min()
            if min_samples < 2:
                # Not enough samples for SMOTE
                return X_train, y_train
            k = min(3, min_samples - 1)
            smote = SMOTE(random_state=42, k_neighbors=max(1, k))
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        return X_train, y_train

    def split_data(self, X, y):
        """Split into train (70%), validation (15%), test (15%)."""
        seed = self.config['dataset']['random_seed']

        # Check if stratification is possible
        class_counts = np.bincount(y.astype(int))
        min_count = class_counts[class_counts > 0].min()

        # First split: need at least 2 in each side -> min 4 total
        stratify_y = y if min_count >= 4 else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=stratify_y
        )

        # Second split: check temp set class distribution
        temp_counts = np.bincount(y_temp.astype(int))
        temp_min = temp_counts[temp_counts > 0].min() if len(temp_counts) > 0 else 0
        stratify_temp = y_temp if temp_min >= 2 else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed, stratify=stratify_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def run_pipeline(self, df, label_type='binary'):
        """
        Execute full preprocessing pipeline.

        Args:
            df: Raw DataFrame from DataLoader
            label_type: 'binary' or 'multiclass'

        Returns:
            dict with X_train, X_val, X_test, y_train, y_val, y_test, metadata
        """
        # 1. Clean
        df = self.clean_data(df)

        # 2. Encode labels
        df = self.encode_labels(df)

        # 3. Select label column
        label_col = 'label_binary' if label_type == 'binary' else 'label_multiclass'

        # 4. Separate features and labels before encoding categoricals
        exclude_cols = ['label', 'Label', 'attack_category', 'split', 'label_binary', 'label_multiclass']
        feature_df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
        y = df[label_col].values

        # 5. Encode categorical features
        feature_df = self.encode_categorical(feature_df)

        # 6. Get feature names and convert to numpy
        feature_names = feature_df.columns.tolist()
        X = feature_df.values.astype(np.float32)

        # 7. Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # 8. Scale
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        # 9. Feature selection
        k = min(self.config['preprocessing']['top_features'], X_train.shape[1])
        X_train = self.select_features(X_train, y_train, feature_names, k=k)
        X_val = self.feature_selector.transform(X_val)
        X_test = self.feature_selector.transform(X_test)

        # 10. Handle imbalance (on training set only)
        X_train, y_train = self.handle_imbalance(X_train, y_train)

        # 11. Save pipeline artifacts
        self._save_artifacts()

        self.feature_names = self.selected_features
        self.n_features = X_train.shape[1]

        return {
            'X_train': X_train.astype(np.float32),
            'X_val': X_val.astype(np.float32),
            'X_test': X_test.astype(np.float32),
            'y_train': y_train.astype(np.int64),
            'y_val': y_val.astype(np.int64),
            'y_test': y_test.astype(np.int64),
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'feature_names': self.selected_features,
            'label_type': label_type,
        }

    def _save_artifacts(self):
        """Save preprocessing artifacts for later reuse."""
        save_dir = resolve_path('models/preprocessing')
        os.makedirs(save_dir, exist_ok=True)

        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, os.path.join(save_dir, 'feature_selector.pkl'))

    def load_artifacts(self):
        """Load saved preprocessing artifacts."""
        save_dir = resolve_path('models/preprocessing')
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))
        self.feature_selector = joblib.load(os.path.join(save_dir, 'feature_selector.pkl'))

    def transform(self, df):
        """Transform new data using fitted pipeline (for inference)."""
        df = self.clean_data(df)
        df = self.encode_categorical(df)

        exclude_cols = ['label', 'attack_category', 'split', 'label_binary', 'label_multiclass']
        feature_df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')

        X = feature_df.values.astype(np.float32)
        X = self.scaler.transform(X)
        X = self.feature_selector.transform(X)
        return X.astype(np.float32)
