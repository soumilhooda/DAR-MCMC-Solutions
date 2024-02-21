import numpy as np
import matplotlib.pyplot as plt

def gaussian_pdf(x, mean, var):
  """
  Calculates the probability density function of a Gaussian distribution.

  Args:
    x: The value at which to evaluate the PDF.
    mean: The mean of the Gaussian distribution.
    var: The variance of the Gaussian distribution.

  Returns:
    The value of the PDF at x.
  """
  return np.exp(-((x - mean)**2) / (2 * var)) / (np.sqrt(2 * np.pi * var))

def metropolis_hastings(pdf, x0, mean, var, num_steps):
  """
  Implements the Metropolis-Hastings algorithm.

  Args:
    pdf: The target probability density function.
    x0: The initial state of the sampler.
    mean: The mean of the Gaussian distribution.
    var: The variance of the Gaussian distribution.
    num_steps: The number of sampling steps.

  Returns:
    A list of samples drawn from the target distribution.
  """
  samples = [x0]
  for _ in range(num_steps):
    x_new = np.random.normal(loc=x0, scale=1)
    alpha = min(1, pdf(x_new, mean, var) / pdf(x0, mean, var))
    if np.random.rand() < alpha:
      x0 = x_new
    samples.append(x0)
  return samples

def main():
  """
  Plots the samples drawn from the MCMC sampler and the true Gaussian distribution.
  """
  mean = 2
  var = 2
  num_steps = 10000

  samples = metropolis_hastings(gaussian_pdf, 0, mean, var, num_steps)

  plt.hist(samples, density=True)
  x = np.linspace(-5, 5, 1000)
  plt.plot(x, gaussian_pdf(x, mean, var), label='True PDF')
  plt.xlabel('x')
  plt.ylabel('Probability density')
  plt.title('MCMC samples vs. True Gaussian distribution')
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()
