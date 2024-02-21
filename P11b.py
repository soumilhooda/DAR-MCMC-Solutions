import numpy as np
import matplotlib.pyplot as plt

def compute_autocorrelation(chain):
    """
    Compute the autocorrelation function of a chain using FFT.

    Args:
        chain (numpy.ndarray): The chain for which to compute the autocorrelation.

    Returns:
        numpy.ndarray: The autocorrelation function.
    """
    n = len(chain)
    chain_padded = np.concatenate((chain, np.zeros(n)))
    fft_result = np.fft.fft(chain_padded)
    autocorrelation = np.real(np.fft.ifft(fft_result * np.conjugate(fft_result)))[:n]
    autocorrelation /= autocorrelation[0]
    return autocorrelation

if __name__ == "__main__":
    # Replace with your actual chain obtained from Problem 2
    np.random.seed(0)
    chain = np.random.normal(2, np.sqrt(2), size=10000)

    autocorrelation = compute_autocorrelation(chain)

    lag_range = 100
    plt.plot(range(lag_range), autocorrelation[:lag_range], marker='o', linestyle='-')
    plt.xlabel('Lag (Δ)')
    plt.ylabel('Autocorrelation')
    plt.title('Empirical Autocorrelation Function (Δ < 100)')
    plt.grid(True)
    plt.show()
