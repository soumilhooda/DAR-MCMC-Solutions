import numpy as np
import matplotlib.pyplot as plt

def generate_samples(num_samples, x_min, x_max, y_min, y_max):
    """
    Generates random samples from a uniform distribution within specified region constraints.

    Args:
        num_samples (int): Number of samples to generate.
        x_min (float): Minimum value of x.
        x_max (float): Maximum value of x.
        y_min (float): Minimum value of y.
        y_max (float): Maximum value of y.

    Returns:
        numpy.ndarray: Array of shape (num_samples, 2) containing the generated samples.
    """
    samples = []
    for _ in range(num_samples):
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(y_min, y_max)
        samples.append([x_sample, y_sample])
    return np.array(samples)

if __name__ == "__main__":
    # Define the region constraints
    x_min = 3
    x_max = 7
    y_min = 1
    y_max = 9
    num_samples = 1000

    # Generate random samples from the top-hat distribution within the specified region
    samples_top_hat = generate_samples(num_samples, x_min, x_max, y_min, y_max)

    # Set samples outside the region to zero (masking)
    mask = ((samples_top_hat[:, 0] >= x_min) & (samples_top_hat[:, 0] <= x_max) &
            (samples_top_hat[:, 1] >= y_min) & (samples_top_hat[:, 1] <= y_max))
    samples_top_hat = samples_top_hat[mask]

    # Plot the histograms
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(samples_top_hat[:, 0], bins=30, density=True, alpha=0.6)
    plt.title('Histogram of x (Top Hat)')
    plt.xlim(x_min, x_max)
    plt.subplot(1, 2, 2)
    plt.hist(samples_top_hat[:, 1], bins=30, density=True, alpha=0.6)
    plt.title('Histogram of y (Top Hat)')
    plt.xlim(y_min, y_max)
    plt.show()

    # Plot the scatterplot
    plt.figure(figsize=(6, 6))
    plt.scatter(samples_top_hat[:, 0], samples_top_hat[:, 1], alpha=0.5)
    plt.xlabel('x (Top Hat)')
    plt.ylabel('y (Top Hat)')
    plt.title('Scatterplot (Top Hat)')
    plt.grid(True)
    plt.show()
