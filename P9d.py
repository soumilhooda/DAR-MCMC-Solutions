import numpy as np
import matplotlib.pyplot as plt

def autocorr(x):
    """
    Compute the autocorrelation of the signal.

    Args:
        x (numpy.ndarray): Input signal.

    Returns:
        numpy.ndarray: Autocorrelation values.
    """
    xp = x - np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size // 2] / np.sum(xp**2)

if __name__ == "__main__":
    # Generate some test data
    N = 10000
    x = np.random.normal(size=N)

    # Compute and plot the autocorrelation function
    acf = autocorr(x)
    plt.plot(acf)
    plt.xlim(0, 100)  # Limit to short lags
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function')
    plt.grid(True)
    plt.show()
