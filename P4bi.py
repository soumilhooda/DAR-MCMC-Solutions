import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_samples(mean, covariance, num_samples):
    """Generates random samples from a multivariate Gaussian distribution.

    Args:
        mean: Mean vector of the Gaussian distribution.
        covariance: Covariance matrix of the Gaussian distribution.
        num_samples: Number of samples to generate.

    Returns:
        A NumPy array of shape (num_samples, 2) containing the samples.
    """
    return np.random.multivariate_normal(mean, covariance, num_samples)

def plot_1d_histograms(samples):
    """Plots 1D histograms of the x and y components of the samples.

    Args:
        samples: A NumPy array of shape (num_samples, 2) containing samples.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(samples[:, 0], bins=30, density=True, alpha=0.6)
    plt.title('Histogram of x')
    plt.subplot(1, 2, 2)
    plt.hist(samples[:, 1], bins=30, density=True, alpha=0.6)
    plt.title('Histogram of y')

def plot_2d_scatterplot(samples):
    """Plots a 2D scatterplot of the samples.

    Args:
        samples: A NumPy array of shape (num_samples, 2) containing samples.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatterplot (Gaussian)')
    plt.grid(True)

if __name__ == "__main__":
    mean = [5, 5]
    covariance = np.array([[2.0, 1.2], [1.2, 2.0]])
    num_samples = 1000

    samples = generate_gaussian_samples(mean, covariance, num_samples)

    plot_1d_histograms(samples)
    plot_2d_scatterplot(samples)

    plt.show()
