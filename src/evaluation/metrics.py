import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Standard attack category names
ATTACK_CATEGORIES = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}


def compute_all_metrics(y_true, y_pred, y_probs=None, average='weighted'):
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional, for AUC-ROC)
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')

    Returns:
        dict of all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    # False positive/negative rates (binary)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        # Multi-class: compute per-class and average
        n_classes = cm.shape[0]
        fprs = []
        fnrs = []
        for i in range(n_classes):
            tp_i = cm[i, i]
            fn_i = cm[i, :].sum() - tp_i
            fp_i = cm[:, i].sum() - tp_i
            tn_i = cm.sum() - tp_i - fn_i - fp_i
            fprs.append(fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0)
            fnrs.append(fn_i / (fn_i + tp_i) if (fn_i + tp_i) > 0 else 0)
        metrics['fpr'] = float(np.mean(fprs))
        metrics['fnr'] = float(np.mean(fnrs))

    # AUC-ROC
    if y_probs is not None:
        try:
            if len(np.unique(y_true)) == 2:
                probs = y_probs[:, 1] if y_probs.ndim > 1 else y_probs
                metrics['auc_roc'] = roc_auc_score(y_true, probs)
            else:
                metrics['auc_roc'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average=average
                )
        except ValueError:
            metrics['auc_roc'] = 0.0

    return metrics


def compute_per_class_metrics(y_true, y_pred, y_probs=None):
    """
    Compute metrics broken down by each attack class.

    Returns:
        dict mapping class_name -> {accuracy, precision, recall, f1_score, support, fpr}
    """
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class = {}

    prec_per = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    for idx, cls in enumerate(classes):
        cls_name = ATTACK_CATEGORIES.get(int(cls), f'Class_{cls}')
        mask_true = (y_true == cls)
        mask_pred = (y_pred == cls)
        support = int(mask_true.sum())

        # Per-class accuracy: correct predictions for this class / total of this class
        cls_correct = int(((y_pred == cls) & (y_true == cls)).sum())
        cls_acc = cls_correct / support if support > 0 else 0.0

        # FPR for this class
        tp_i = cm[idx, idx]
        fp_i = cm[:, idx].sum() - tp_i
        fn_i = cm[idx, :].sum() - tp_i
        tn_i = cm.sum() - tp_i - fn_i - fp_i
        fpr_i = fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0.0

        per_class[cls_name] = {
            'class_id': int(cls),
            'precision': float(prec_per[idx]),
            'recall': float(rec_per[idx]),
            'f1_score': float(f1_per[idx]),
            'support': support,
            'accuracy': float(cls_acc),
            'fpr': float(fpr_i),
        }

    return per_class


def compute_robust_accuracy(y_true, y_pred_clean, y_pred_adv):
    """
    Compute clean and robust accuracy.

    Args:
        y_true: True labels
        y_pred_clean: Predictions on clean data
        y_pred_adv: Predictions on adversarial data
    """
    clean_acc = accuracy_score(y_true, y_pred_clean)
    robust_acc = accuracy_score(y_true, y_pred_adv)
    accuracy_drop = clean_acc - robust_acc

    return {
        'clean_accuracy': clean_acc,
        'robust_accuracy': robust_acc,
        'accuracy_drop': accuracy_drop,
        'robustness_ratio': robust_acc / clean_acc if clean_acc > 0 else 0
    }


def compute_per_class_robust_metrics(y_true, y_pred_clean, y_pred_adv):
    """
    Compute per-class robust accuracy breakdown.

    Returns:
        dict mapping class_name -> {clean_accuracy, robust_accuracy, accuracy_drop, support}
    """
    classes = sorted(np.unique(y_true))
    per_class = {}

    for cls in classes:
        cls_name = ATTACK_CATEGORIES.get(int(cls), f'Class_{cls}')
        mask = (y_true == cls)
        support = int(mask.sum())
        if support == 0:
            continue

        clean_acc = float(accuracy_score(y_true[mask], y_pred_clean[mask]))
        robust_acc = float(accuracy_score(y_true[mask], y_pred_adv[mask]))

        per_class[cls_name] = {
            'class_id': int(cls),
            'clean_accuracy': clean_acc,
            'robust_accuracy': robust_acc,
            'accuracy_drop': clean_acc - robust_acc,
            'robustness_ratio': robust_acc / clean_acc if clean_acc > 0 else 0,
            'support': support,
        }

    return per_class
