import torch
import torch.nn.functional as F


def pgd_attack(model, x, y, epsilon=0.1, alpha=0.01, num_iterations=40, random_start=True):
    """
    Projected Gradient Descent (Madry et al., 2018)

    Iteratively applies:
        x_adv = x_adv + alpha * sign(gradient)
        x_adv = project(x_adv, x, epsilon)

    Args:
        model: PyTorch model
        x: input tensor
        y: true labels
        epsilon: max perturbation magnitude
        alpha: step size per iteration
        num_iterations: number of attack steps
        random_start: initialize with random perturbation

    Returns:
        x_adv: adversarial examples
    """
    x_adv = x.clone().detach()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(num_iterations):
        x_adv.requires_grad_(True)

        output = model(x_adv)
        loss = F.cross_entropy(output, y)

        loss.backward()
        grad_sign = x_adv.grad.sign()

        # Take step
        x_adv = x_adv.detach() + alpha * grad_sign

        # Project back into epsilon-ball around original x
        perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + perturbation, 0, 1)

    return x_adv.detach()
