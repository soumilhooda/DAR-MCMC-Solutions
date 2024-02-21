import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x, mean, covariance_matrix):
    """
    Computes the probability density function of the target distribution at a given point x.

    Args:
        x (numpy.ndarray): Point at which to evaluate the PDF.
        mean (numpy.ndarray): Mean vector of the Gaussian distribution.
        covariance_matrix (numpy.ndarray): Covariance matrix of the Gaussian distribution.

    Returns:
        float: Probability density function value at point x.
    """
    return np.exp(-0.5 * (x - mean).T @ np.linalg.inv(covariance_matrix) @ (x - mean))

def metropolis_hastings_shifted_mean(initial_state, num_samples, proposal_variance, mean, covariance_matrix):
    """
    Runs the Metropolis-Hastings MCMC algorithm with a shifted mean proposal.

    Args:
        initial_state (numpy.ndarray): Initial state of the Markov chain.
        num_samples (int): Number of samples to generate.
        proposal_variance (float): Variance of the Gaussian proposal distribution.
        mean (numpy.ndarray): Mean vector of the Gaussian target distribution.
        covariance_matrix (numpy.ndarray): Covariance matrix of the Gaussian target distribution.

    Returns:
        numpy.ndarray: Array of generated samples.
    """
    current_state = initial_state
    samples = [current_state]

    for _ in range(num_samples):
        # Propose a new sample using a Gaussian proposal distribution with shifted mean
        proposal_mean = current_state + 2  # Shifted mean
        proposal = np.random.multivariate_normal(proposal_mean, proposal_variance * np.eye(2))

        # Calculate acceptance ratio using the shifted proposal mean
        acceptance_ratio = min(1, target_distribution(proposal, mean, covariance_matrix) /
                                    target_distribution(current_state, mean, covariance_matrix))

        # Accept or reject the proposed sample
        if np.random.rand() < acceptance_ratio:
            current_state = proposal
        samples.append(current_state)

    return np.array(samples)

if __name__ == "__main__":
    # Define the parameters for the Gaussian distribution
    mean = np.array([5, 5])
    covariance_matrix = np.array([[2, 0], [0, 3]])

    # Initial state
    initial_state = np.array([1, 1])

    # Number of samples to generate
    num_samples = 1000

    # Proposal Variance
    proposal_variance = 1

    # Run the Metropolis-Hastings algorithm with shifted mean
    samples = metropolis_hastings_shifted_mean(initial_state, num_samples, proposal_variance, mean, covariance_matrix)

    # Plot the samples
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title('Metropolis-Hastings with Shifted Mean Proposal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.show()
