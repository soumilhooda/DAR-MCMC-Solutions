import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def mcmc_sampling(Q, num_samples=10000):
    """
    Perform MCMC sampling using the Metropolis-Hastings algorithm.

    Args:
        Q (float): The proposal variance.
        num_samples (int, optional): Number of samples to generate. Default is 10000.

    Returns:
        numpy.ndarray: The generated chain of samples.
        float: The acceptance fraction.
    """
    # Define the target density
    def target_density(x):
        return stats.multivariate_normal(mean=[0, 0], cov=[[2.0, 1.2], [1.2, 2.0]]).pdf(x)

    # Initialize the chain
    chain = np.zeros((num_samples, 2))
    chain[0, :] = np.random.randn(2)

    # Run the MCMC sampling
    num_accepted = 0
    for i in range(1, num_samples):
        # Generate a proposal sample
        proposal = chain[i-1, :] + np.sqrt(Q) * np.random.randn(2)

        # Compute the acceptance probability
        p_accept = min(1, target_density(proposal) / target_density(chain[i-1, :]))

        # Accept or reject the proposal
        if np.random.rand() < p_accept:
            chain[i, :] = proposal
            num_accepted += 1
        else:
            chain[i, :] = chain[i-1, :]

    # Return the chain and acceptance fraction
    return chain, num_accepted / num_samples

# Evaluate the acceptance fraction for different values of Q
Q_values = 2.0**np.arange(-10, 10)

acceptance_fractions = []
for Q in Q_values:
    _, acceptance_fraction = mcmc_sampling(Q)
    acceptance_fractions.append(acceptance_fraction)

# Plot the acceptance fraction as a function of Q
plt.plot(Q_values, acceptance_fractions)
plt.xscale('log')
plt.xlabel('Q')
plt.ylabel('Acceptance fraction')
plt.show()

# Find the value of Q that gives an acceptance fraction of about 0.25
Q_optimal = Q_values[np.argmin(np.abs(np.array(acceptance_fractions) - 0.25))]
print(f'Optimal Q: {Q_optimal}')
