import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.utils.config import resolve_path


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_roc_curves(y_true, y_probs, class_names, save_path=None):
    """Plot ROC curves for multi-class classification."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n_classes = y_probs.shape[1] if y_probs.ndim > 1 else 2

    if n_classes == 2:
        probs = y_probs[:, 1] if y_probs.ndim > 1 else y_probs
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(n_classes))
        for i in range(n_classes):
            if y_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                name = class_names[i] if i < len(class_names) else f'Class {i}'
                ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_epsilon_sensitivity(epsilons, accuracies, attack_names, save_path=None):
    """Plot accuracy vs epsilon for adversarial robustness."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, accs in zip(attack_names, accuracies):
        ax.plot(epsilons, accs, marker='o', label=name)

    ax.set_xlabel('Epsilon (Perturbation Magnitude)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Adversarial Robustness: Accuracy vs Epsilon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_tier_breakdown(tier_counts, save_path=None):
    """Plot tier-wise detection contribution pie chart."""
    labels = [f'Tier {k}' for k in tier_counts.keys()]
    sizes = list(tier_counts.values())

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%', startangle=90)
    ax.set_title('Detection Contribution by Tier')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_training_history(history, title='Training History', save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.get('loss', []), label='Train Loss')
    ax1.plot(history.get('val_loss', []), label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.get('accuracy', []), label='Train Accuracy')
    ax2.plot(history.get('val_accuracy', []), label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_baseline_comparison(comparison_data, save_path=None):
    """Plot bar chart comparing different system configurations."""
    import pandas as pd

    df = pd.DataFrame(comparison_data)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.2

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax.bar(x + i * width, df[metric], width, label=metric)

    ax.set_xlabel('System Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Baseline System Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['System'], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_feature_importance(feature_names, importances, top_k=20, save_path=None):
    """Plot feature importance bar chart."""
    # Sort and take top-k
    sorted_idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_k} Feature Importances')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig
