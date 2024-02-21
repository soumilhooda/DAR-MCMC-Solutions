import numpy as np
import matplotlib.pyplot as plt

def target_density(x):
    """
    Compute the probability density function (PDF) of the target distribution.

    Args:
        x (float or numpy.ndarray): Input value(s) for which to compute the PDF.

    Returns:
        float or numpy.ndarray: PDF value(s) corresponding to the input.
    """
    return (1 / np.sqrt(2 * np.pi * 2)) * np.exp(-(x - 2)**2 / (2 * 2))

def proposal_distribution(x, sigma=1.0):
    """
    Generate samples from the proposal distribution.

    Args:
        x (float or numpy.ndarray): Current state(s) from which to generate proposals.
        sigma (float): Standard deviation of the proposal distribution.

    Returns:
        float or numpy.ndarray: Sample(s) generated from the proposal distribution.
    """
    return np.random.normal(x, sigma)

if __name__ == "__main__":
    num_samples = 10000
    samples = np.zeros(num_samples)
    x = 0
    accept_count = 0

    for i in range(num_samples):
        x_candidate = proposal_distribution(x)
        acceptance_ratio = target_density(x_candidate) / target_density(x)

        if np.random.rand() < acceptance_ratio:
            x = x_candidate
            accept_count += 1

        samples[i] = x

    acceptance_rate = accept_count / num_samples
    print(f"Acceptance rate: {acceptance_rate:.2%}")

    plt.hist(samples, bins=50, density=True, alpha=0.6, label='MCMC Samples')
    x_values = np.linspace(-2, 6, 400)
    plt.plot(x_values, target_density(x_values), 'r-', label='True Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Metropolis-Hastings MCMC Sampling')
    plt.show()
