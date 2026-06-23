import numpy as np
import torch


def deepfool_attack(model, x, input_shape, num_classes, max_iter=50, epsilon=1e-6):
    """
    DeepFool Attack (Moosavi-Dezfooli et al., 2016)

    Iteratively finds the closest decision boundary
    and computes the minimal perturbation to cross it.

    Uses ART library for implementation.

    Args:
        model: PyTorch model
        x: input numpy array
        input_shape: shape of single input
        num_classes: number of output classes
        max_iter: max iterations
        epsilon: overshoot to ensure crossing boundary

    Returns:
        x_adv: adversarial examples (numpy array)
    """
    try:
        from art.attacks.evasion import DeepFool as DeepFoolART
        from art.estimators.classification import PyTorchClassifier

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        art_classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=num_classes,
            clip_values=(0.0, 1.0)
        )

        attack = DeepFoolART(
            classifier=art_classifier,
            max_iter=max_iter,
            epsilon=epsilon,
            nb_grads=num_classes
        )

        x_adv = attack.generate(x=x.astype(np.float32))
        return x_adv

    except ImportError:
        print("Warning: ART library not available. Returning original inputs.")
        return x.copy()
