import numpy as np
import matplotlib.pyplot as plt

# Define the target density function (constant density, i.e., unity everywhere)
def target_density(x):
    return 1.0

# Metropolis-Hastings MCMC sampler
def metropolis_hastings_sampler(initial_x, num_steps, proposal_std):
    current_x = initial_x
    samples = [current_x]

    for _ in range(num_steps):
        # Propose a new sample from the proposal distribution (Gaussian)
        proposal_x = np.random.normal(current_x, proposal_std)

        # Calculate the acceptance ratio
        acceptance_ratio = target_density(proposal_x) / target_density(current_x)

        # Accept or reject the proposed sample
        if np.random.rand() < acceptance_ratio:
            current_x = proposal_x
        samples.append(current_x)

    return np.array(samples)

# Initial state and parameters
initial_x = 0.0  # Initial value of x
num_steps = 50000  # Number of MCMC steps
proposal_std = 1.0  # Standard deviation of the proposal distribution

# Run the Metropolis-Hastings sampler
samples = metropolis_hastings_sampler(initial_x, num_steps, proposal_std)

# Plot the chain value x as a function of time step
plt.figure(figsize=(10, 4))
plt.plot(samples)
plt.xlabel('Time Step')
plt.ylabel('x')
plt.title('Metropolis-Hastings MCMC Sampler with Unity Density')
plt.grid(True)
plt.show()
