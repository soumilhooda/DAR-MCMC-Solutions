import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def target_distribution(x, mean, std):
    """
    Compute the probability density function (PDF) of the target distribution.

    Args:
        x (float or numpy.ndarray): Input value(s) for which to compute the PDF.
        mean (float): Mean of the target distribution.
        std (float): Standard deviation of the target distribution.

    Returns:
        float or numpy.ndarray: PDF value(s) corresponding to the input.
    """
    return stats.norm.pdf(x, loc=mean, scale=std)

def proposal_distribution(x, sigma=1.0):
    """
    Generate samples from the proposal distribution.

    Args:
        x (float or numpy.ndarray): Current state(s) from which to generate proposals.
        sigma (float): Standard deviation of the proposal distribution.

    Returns:
        float or numpy.ndarray: Sample(s) generated from the proposal distribution.
    """
    return np.random.normal(loc=x, scale=sigma)

if __name__ == "__main__":
    np.random.seed(0)
    target_mean = 2
    target_std = 2
    rv = stats.norm(loc=target_mean, scale=target_std)
    x = 0
    chain = [x]
    num_steps = 10000

    for i in range(num_steps):
        x_prime = proposal_distribution(x)
        acceptance_probability = min(1, rv.pdf(x_prime) / rv.pdf(x))
        
        if np.random.uniform() < acceptance_probability:
            x = x_prime
        
        chain.append(x)

    plt.plot(range(num_steps + 1), chain)
    plt.xlabel('Time Step')
    plt.ylabel('x')
    plt.title('Chain of x Values vs. Time Step')
    plt.grid(True)
    plt.show()
