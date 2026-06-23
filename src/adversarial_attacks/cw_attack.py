import numpy as np
import torch


def cw_attack(model, x, y, input_shape, num_classes,
              confidence=0.0, max_iter=100, learning_rate=0.01):
    """
    Carlini & Wagner L2 Attack (Carlini & Wagner, 2017)

    Uses ART library for implementation.

    Args:
        model: PyTorch model
        x: input numpy array
        y: true labels numpy array
        input_shape: shape of single input
        num_classes: number of output classes
        confidence: attack confidence parameter
        max_iter: maximum optimization iterations
        learning_rate: optimizer step size

    Returns:
        x_adv: adversarial examples (numpy array)
    """
    try:
        from art.attacks.evasion import CarliniL2Method
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

        attack = CarliniL2Method(
            classifier=art_classifier,
            confidence=confidence,
            max_iter=max_iter,
            learning_rate=learning_rate,
            binary_search_steps=5,
            initial_const=0.01
        )

        x_adv = attack.generate(x=x.astype(np.float32))
        return x_adv

    except ImportError:
        print("Warning: ART library not available. Returning original inputs.")
        return x.copy()
