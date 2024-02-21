import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_proposal(current_state, proposal_cov):
  """Generates a new state proposed using a Gaussian distribution.

  Args:
    current_state:  The current position in the parameter space.
    proposal_cov: Covariance matrix of the proposal distribution.

  Returns:
    The proposed new state.
  """
  return np.random.multivariate_normal(mean=current_state, cov=proposal_cov)

def target_density(x, mu, cov):
  """Calculates the target density using a multivariate Gaussian distribution.

  Args:
    x: The position in the parameter space where to evaluate the density.
    mu: Mean vector of the multivariate Gaussian distribution.
    cov: Covariance matrix of the multivariate Gaussian distribution.

  Returns:
    The target density value at position x.
  """
  return multivariate_normal.pdf(x, mean=mu, cov=cov)

def metropolis_hastings_sampler(target_density, initial_state, proposal_cov, num_steps):
  """Implements the Metropolis-Hastings MCMC sampler.

  Args:
    target_density: The target probability density function.
    initial_state: The initial state of the sampler.
    proposal_cov: Covariance matrix of the proposal distribution.
    num_steps: Number of sampling steps.

  Returns:
    A list of samples drawn from the target distribution.
  """
  samples = [initial_state]
  current_state = initial_state

  for _ in range(num_steps):
    proposed_state = generate_proposal(current_state, proposal_cov)
    alpha = min(1, target_density(proposed_state, mu, cov) / target_density(current_state, mu, cov))

    if np.random.rand() < alpha:
      current_state = proposed_state

    samples.append(current_state.copy())  # Store a copy

  return samples

# Main script
if __name__ == "__main__":
  mu = np.array([0, 0])
  cov = np.array([[2.0, 1.2], [1.2, 2.0]])
  proposal_cov = np.eye(2)
  initial_state = np.array([0, 0])
  num_steps = 10000

  samples = metropolis_hastings_sampler(target_density, initial_state, proposal_cov, num_steps)

  # Plot results (same as before) 
  plt.scatter(np.array(samples)[:, 0], np.array(samples)[:, 1], alpha=0.5)
  plt.show()
