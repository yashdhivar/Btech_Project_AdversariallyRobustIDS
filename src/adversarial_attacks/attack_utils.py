import numpy as np
import torch
import torch.nn as nn


class PyTorchDNN(nn.Module):
    """
    PyTorch mirror of the Keras DNN for adversarial attack generation.
    Provides gradient computation needed for FGSM/PGD attacks.
    """

    def __init__(self, input_dim, num_classes):
        super(PyTorchDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def evaluate_attack(model, x_clean, x_adv, y_true):
    """
    Evaluate adversarial attack effectiveness.

    Returns:
        dict with attack_success_rate, perturbation metrics
    """
    model.eval()

    if isinstance(x_clean, np.ndarray):
        x_clean_t = torch.FloatTensor(x_clean)
        x_adv_t = torch.FloatTensor(x_adv)
    else:
        x_clean_t = x_clean
        x_adv_t = x_adv

    if isinstance(y_true, np.ndarray):
        y_true_np = y_true
    else:
        y_true_np = y_true.numpy()

    with torch.no_grad():
        pred_clean = model(x_clean_t).argmax(dim=1).numpy()
        pred_adv = model(x_adv_t).argmax(dim=1).numpy()

    # Attack success = prediction changed from correct to incorrect
    correctly_classified = pred_clean == y_true_np
    misclassified_after = pred_adv != y_true_np

    num_correct = correctly_classified.sum()
    if num_correct == 0:
        attack_success_rate = 0.0
    else:
        attack_success = (correctly_classified & misclassified_after).sum()
        attack_success_rate = float(attack_success / num_correct * 100)

    # Perturbation metrics
    if isinstance(x_adv, torch.Tensor):
        x_adv_np = x_adv.numpy()
        x_clean_np = x_clean.numpy()
    else:
        x_adv_np = x_adv
        x_clean_np = x_clean

    perturbation = x_adv_np - x_clean_np
    avg_l2 = float(np.mean(np.linalg.norm(perturbation, axis=1)))
    avg_linf = float(np.mean(np.max(np.abs(perturbation), axis=1)))

    return {
        'attack_success_rate': attack_success_rate,
        'avg_l2_perturbation': avg_l2,
        'avg_linf_perturbation': avg_linf,
        'samples_attacked': int(num_correct),
        'samples_fooled': int((correctly_classified & misclassified_after).sum()) if num_correct > 0 else 0
    }


def generate_mixed_adversarial_dataset(model, x, y, config):
    """
    Generate a mixed adversarial dataset using multiple attack methods.
    """
    from src.adversarial_attacks.fgsm import fgsm_attack
    from src.adversarial_attacks.pgd import pgd_attack

    n = len(x)
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.LongTensor(y)

    # Split indices for each attack type
    indices = np.random.permutation(n)
    fgsm_idx = indices[:int(0.3 * n)]
    pgd_idx = indices[int(0.3 * n):int(0.6 * n)]
    cw_idx = indices[int(0.6 * n):int(0.8 * n)]
    df_idx = indices[int(0.8 * n):]

    x_adv = np.copy(x)

    # FGSM samples
    if len(fgsm_idx) > 0:
        x_adv[fgsm_idx] = fgsm_attack(
            model, x_tensor[fgsm_idx], y_tensor[fgsm_idx],
            epsilon=config['adversarial_attacks']['fgsm']['epsilon']
        ).numpy()

    # PGD samples
    if len(pgd_idx) > 0:
        x_adv[pgd_idx] = pgd_attack(
            model, x_tensor[pgd_idx], y_tensor[pgd_idx],
            epsilon=config['adversarial_attacks']['pgd']['epsilon'],
            alpha=config['adversarial_attacks']['pgd']['alpha'],
            num_iterations=min(config['adversarial_attacks']['pgd']['num_iterations'], 10)
        ).numpy()

    # Labels: mark which attack generated each sample
    attack_labels = np.zeros(n, dtype=int)
    attack_labels[fgsm_idx] = 1   # FGSM
    attack_labels[pgd_idx] = 2    # PGD
    attack_labels[cw_idx] = 3     # C&W (placeholder - not generated for speed)
    attack_labels[df_idx] = 4     # DeepFool (placeholder)

    return x_adv, attack_labels
