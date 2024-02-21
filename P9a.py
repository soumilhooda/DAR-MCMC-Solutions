import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def target_distribution_ln(ln_x):
    """
    Computes the value of the target density function in ln(x) space.

    Args:
        ln_x (float): Value of ln(x) at which to evaluate the density function.

    Returns:
        float: Value of the target density function at ln(x).
    """
    x = np.exp(ln_x)
    return norm.pdf(ln_x, loc=np.log(2), scale=np.sqrt(2)) / x

def proposal_distribution_ln(ln_x):
    """
    Generates a candidate sample from the proposal distribution in ln(x) space.

    Args:
        ln_x (float): Current value of ln(x).

    Returns:
        float: Candidate sample value of ln(x).
    """
    return np.random.normal(loc=ln_x, scale=1)

if __name__ == "__main__":
    np.random.seed(0)

    ln_x = np.log(1.0)  # Initial value of ln(x)
    samples_ln = [ln_x]

    for i in range(100000):
        ln_x_prime = proposal_distribution_ln(ln_x)
        acceptance_probability = min(1, target_distribution_ln(ln_x_prime) / target_distribution_ln(ln_x))

        if np.random.uniform() < acceptance_probability:
            ln_x = ln_x_prime
        samples_ln.append(ln_x)

    samples_x = np.exp(samples_ln)

    ln_x_values = np.linspace(0, 4, 10000)
    density_ln = target_distribution_ln(ln_x_values)

    plt.hist(samples_x, bins=50, density=True, label='Samples')
    plt.plot(np.exp(ln_x_values), density_ln, label='True density (ln(x))', color='red')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Metropolis-Hastings MCMC Sampler with Steps in ln(x) (Range from 0)')
    plt.legend()
    plt.grid(True)
    plt.show()
