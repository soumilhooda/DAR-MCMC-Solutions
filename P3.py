import numpy as np
import matplotlib.pyplot as plt

def uniform_pdf(x, a, b):
    """
    Calculates the probability density function of a uniform distribution.

    Args:
      x: The value at which to evaluate the PDF.
      a: The lower bound of the uniform distribution.
      b: The upper bound of the uniform distribution.

    Returns:
      The value of the PDF at x.
    """
    result = np.zeros_like(x)  # Create an array of zeros with the same shape as x
    result[(a < x) & (x < b)] = 1 / (b - a)  # Set elements within bounds to the PDF value
    return result


def metropolis_hastings(pdf, x0, a, b, num_steps):
  """
  Implements the Metropolis-Hastings algorithm.

  Args:
    pdf: The target probability density function.
    x0: The initial state of the sampler.
    a: The lower bound of the uniform distribution.
    b: The upper bound of the uniform distribution.
    num_steps: The number of sampling steps.

  Returns:
    A list of samples drawn from the target distribution.
  """
  samples = [x0]
  for _ in range(num_steps):
    x_new = np.random.uniform(a, b)
    alpha = min(1, pdf(x_new, a, b) / pdf(x0, a, b))
    if np.random.rand() < alpha:
      x0 = x_new
    samples.append(x0)
  return samples

def main():
  """
  Plots the samples drawn from the MCMC sampler and the true uniform distribution.
  """
  a = 3
  b = 7
  num_steps = 10000

  samples = metropolis_hastings(uniform_pdf, (a + b) / 2, a, b, num_steps)

  plt.hist(samples, density=True)
  x = np.linspace(a, b, 1000)
  plt.plot(x, uniform_pdf(x, a, b), label='True PDF')
  plt.xlabel('x')
  plt.ylabel('Probability density')
  plt.title('MCMC samples vs. True uniform distribution')
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()
