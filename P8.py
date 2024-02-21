import numpy as np
import matplotlib.pyplot as plt

def target_density(x):
    """
    Computes the value of the target density function at a given point x.

    Args:
        x (float): Point at which to evaluate the density function.

    Returns:
        float: Value of the density function at point x.
    """
    return 1.0

def proposal_distribution(x, sigma=1):
    """
    Generates a candidate sample from the proposal distribution q(x' | x).

    Args:
        x (float): Current sample value.
        sigma (float): Standard deviation of the proposal distribution.

    Returns:
        float: Candidate sample value.
    """
    return np.random.normal(x, sigma)

if __name__ == "__main__":
    # Initialize variables
    num_steps = 10000
    x_values = np.zeros(num_steps)
    x = 0  # Initial value of x

    # Metropolis-Hastings MCMC sampling loop
    for i in range(num_steps):
        x_candidate = proposal_distribution(x)

        # Always accept the candidate (acceptance ratio is 1)
        x = x_candidate

        # Store the current sample
        x_values[i] = x

    # Plot the chain value x as a function of the time step
    plt.plot(range(num_steps), x_values)
    plt.xlabel('Time Step')
    plt.ylabel('Chain Value (x)')
    plt.title('Metropolis-Hastings MCMC Sampling with Constant Density')
    plt.show()
