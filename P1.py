import numpy as np
import matplotlib.pyplot as plt

def estimate_moments(k):
  """
  Estimates the mean, variance, skewness, and kurtosis of a uniform distribution using k samples.

  Args:
    k: The number of samples to draw.

  Returns:
    A tuple containing the estimated mean, variance, skewness, and kurtosis.
  """
  samples = np.random.uniform(0, 1, k)
  mean = np.mean(samples)
  variance = np.var(samples)
  skewness = ((samples - mean)**3).mean() / ((k - 1) * variance**1.5)
  kurtosis = ((samples - mean)**4).mean() / ((k - 1) * variance**2) - 3
  return mean, variance, skewness, kurtosis

def main():
  """
  Plots the estimated mean, variance, skewness, and kurtosis of a uniform distribution as a function of the number of samples.
  """
  num_samples = np.array([4**n for n in range(1, 11)])
  true_mean = 0.5
  true_variance = 1 / 12
  true_skewness = 0
  true_kurtosis = 0

  estimated_means, estimated_variances, estimated_skewnesses, estimated_kurtoses = [], [], [], []
  for k in num_samples:
    mean, variance, skewness, kurtosis = estimate_moments(k)
    estimated_means.append(mean)
    estimated_variances.append(variance)
    estimated_skewnesses.append(skewness)
    estimated_kurtoses.append(kurtosis)

  plt.figure(figsize=(12, 8))

  plt.subplot(2, 2, 1)
  plt.plot(num_samples, estimated_means, label='Estimated Mean', marker='o', linestyle='-')
  plt.plot(num_samples, [true_mean] * len(num_samples), label='True Mean', marker='s', linestyle='--')
  plt.xlabel('Number of Samples (log scale)')
  plt.ylabel('Mean')
  plt.title('Mean')
  plt.legend()
  plt.xscale('log')

  plt.subplot(2, 2, 2)
  plt.plot(num_samples, estimated_variances, label='Estimated Variance', marker='o', linestyle='-')
  plt.plot(num_samples, [true_variance] * len(num_samples), label='True Variance', marker='s', linestyle='--')
  plt.xlabel('Number of Samples (log scale)')
  plt.ylabel('Variance')
  plt.title('Variance')
  plt.legend()
  plt.xscale('log')

  plt.subplot(2, 2, 3)
  plt.plot(num_samples, estimated_skewnesses, label='Estimated Skewness', marker='o', linestyle='-')
  plt.plot(num_samples, [true_skewness] * len(num_samples), label='True Skewness', marker='s', linestyle='--')
  plt.xlabel('Number of Samples (log scale)')
  plt.ylabel('Skewness')
  plt.title('Skewness')
  plt.legend()
  plt.xscale('log')

  plt.subplot(2, 2, 4)
  plt.plot(num_samples, estimated_kurtoses, label='Estimated Kurtosis', marker='o', linestyle='-')
  plt.plot(num_samples, [true_kurtosis] * len(num_samples), label='True Kurtosis', marker='s', linestyle='--')
  plt.xlabel('Number of Samples (log scale)')
  plt.ylabel('Kurtosis')
  plt.title('Kurtosis')
  plt.legend()
  plt.xscale('log')

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
