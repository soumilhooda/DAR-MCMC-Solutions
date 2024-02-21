from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt

def proposal_distribution_ln_x(ln_x, sigma=1.0):
    """
    Generates a candidate sample from the proposal distribution in ln(x) space.

    Args:
        ln_x (float): Current value of ln(x).
        sigma (float): Standard deviation of the proposal distribution.

    Returns:
        float: Candidate sample value of ln(x).
    """
    ln_x_prime = np.random.normal(ln_x, sigma)
    return ln_x_prime

if __name__ == "__main__":
    mean_ln_x = np.log(2)  # Mean of ln(x)
    sigma_ln_x = np.sqrt(np.log(2))  # Standard deviation of ln(x)
    target_distribution = lognorm(scale=np.exp(mean_ln_x), s=sigma_ln_x)

    num_samples = 100000
    samples_ln_x = []
    ln_x = np.log(2)

    for step in range(num_samples):
        ln_x_prime = proposal_distribution_ln_x(ln_x)
        alpha = min(1, target_distribution.pdf(np.exp(ln_x_prime)) / target_distribution.pdf(np.exp(ln_x)))
        if np.random.rand() < alpha:
            ln_x = ln_x_prime
        samples_ln_x.append(ln_x)

    samples_x = np.exp(samples_ln_x)

    plt.hist(samples_x, bins=50, density=True, alpha=0.5, label='MCMC Samples')
    x_range = np.linspace(0, 10, 1000)
    true_density = target_distribution.pdf(x_range)
    plt.plot(x_range, true_density, 'r', label='True Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Metropolis-Hastings MCMC Sampler (ln(x))')
    plt.show()
