import numpy as np
import matplotlib.pyplot as plt

def target_density(x):
    """
    Compute the target density function p(x).

    Args:
        x (float or numpy.ndarray): Input value(s) for which to compute the density.

    Returns:
        float or numpy.ndarray: Density value(s) corresponding to the input.
    """
    return (1 / np.sqrt(2 * np.pi * 2)) * np.exp(-(x - 2)**2 / (2 * 2))

def proposal_distribution(x, sigma=1.0):
    """
    Generate samples from the proposal distribution q(x' | x).

    Args:
        x (float or numpy.ndarray): Current state(s) from which to generate proposals.
        sigma (float): Standard deviation of the Gaussian proposal distribution.

    Returns:
        float or numpy.ndarray: Sample(s) generated from the proposal distribution.
    """
    return np.random.normal(x, sigma)

if __name__ == "__main__":
    num_samples = 10000
    samples_x = np.zeros(num_samples)
    x = 0
    accept_count = 0

    for i in range(num_samples):
        x_candidate = proposal_distribution(x)
        acceptance_ratio = target_density(x_candidate) / target_density(x)

        if np.random.rand() < acceptance_ratio:
            x = x_candidate
            accept_count += 1

        samples_x[i] = x

    acceptance_rate = accept_count / num_samples
    print(f"Acceptance rate: {acceptance_rate:.2%}")

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(range(num_samples), samples_x)
    plt.xlabel('Time Step')
    plt.ylabel('Chain Value (x)')
    plt.title('Chain Value vs. Time Step')

    quarter = num_samples // 4
    segments_x = np.split(samples_x, [quarter, 2 * quarter, 3 * quarter])

    for i, segment in enumerate(segments_x):
        plt.subplot(2, 2, i + 1)
        plt.hist(segment, bins=30, density=True, alpha=0.6, label='Segment Samples')
        mean = np.mean(segment)
        variance = np.var(segment)
        plt.axvline(x=mean, color='r', linestyle='--', label=f'Empirical Mean = {mean:.2f}')
        plt.axvline(x=variance, color='g', linestyle='--', label=f'Empirical Variance = {variance:.2f}')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'Segment {i + 1}')

    plt.tight_layout()
    plt.show()
