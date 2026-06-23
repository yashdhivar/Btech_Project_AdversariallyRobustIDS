import numpy as np


def bit_depth_reduction(x, depth=4):
    """
    Reduce bit depth to remove fine-grained perturbations.
    Maps continuous values to discrete levels.

    depth=4 means 2^4 = 16 levels
    """
    levels = 2 ** depth
    x_transformed = np.round(x * levels) / levels
    return np.clip(x_transformed, 0, 1)


def gaussian_smoothing(x, sigma=0.1):
    """
    Add Gaussian noise to mask adversarial perturbations.
    Small sigma preserves features while disrupting precise perturbations.
    """
    noise = np.random.normal(0, sigma, x.shape)
    x_smoothed = x + noise
    return np.clip(x_smoothed, 0, 1)


def feature_squeezing(x, bit_depth=4):
    """
    Combine multiple transformations.
    Returns squeezed input for comparison with original.
    """
    x_squeezed = bit_depth_reduction(x, depth=bit_depth)
    return x_squeezed
