import torch
import torch.nn.functional as F


def fgsm_attack(model, x, y, epsilon=0.1):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014)

    x_adv = x + epsilon * sign(gradient_of_loss_wrt_x)

    Args:
        model: PyTorch model
        x: input tensor (batch_size, features)
        y: true labels tensor
        epsilon: perturbation magnitude (0.01 to 0.3)

    Returns:
        x_adv: adversarial examples
    """
    x_adv = x.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(x_adv)
    loss = F.cross_entropy(output, y)

    # Compute gradient
    loss.backward()

    # Create perturbation
    perturbation = epsilon * x_adv.grad.sign()

    # Apply perturbation
    x_adv = x + perturbation

    # Clamp to valid range
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()
