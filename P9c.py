import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def target_distribution(x):
    """
    Defines the target distribution.

    Args:
        x (float): The input value.

    Returns:
        float: Probability density function value at x.
    """
    return norm.pdf(x, loc=2, scale=np.sqrt(2))

def proposal_distribution(x):
    """
    Defines the proposal distribution.

    Args:
        x (float): The input value.

    Returns:
        float: Sample from the proposal distribution.
    """
    return np.random.normal(loc=x, scale=1)

if __name__ == "__main__":
    np.random.seed(0)

    x = 0
    samples = [x]

    num_steps = 10000

    for i in range(num_steps):
        x_prime = proposal_distribution(x)
        acceptance_probability = min(1, target_distribution(x_prime) / target_distribution(x))
        if np.random.uniform() < acceptance_probability:
            x = x_prime
        samples.append(x)

    plt.figure(figsize=(10, 4))
    plt.plot(samples)
    plt.xlabel('Time Step')
    plt.ylabel('x')
    plt.title('Chain of x Values over Time Steps')
    plt.grid(True)
    plt.show()

    num_segments = 4
    segment_length = len(samples) // num_segments
    segment_means = []
    segment_variances = []

    for i in range(num_segments):
        segment = samples[i * segment_length:(i + 1) * segment_length]
        segment_means.append(np.mean(segment))
        segment_variances.append(np.var(segment))

    print("Segment Means:", segment_means)
    print("Segment Variances:", segment_variances)
