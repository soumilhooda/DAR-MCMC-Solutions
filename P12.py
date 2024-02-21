import numpy as np
import matplotlib.pyplot as plt

def autocorr_func_1d(x, norm=True):
    """
    Compute the 1D autocorrelation function of a signal.

    Args:
        x (numpy.ndarray): The signal.
        norm (bool): Whether to normalize the autocorrelation function.

    Returns:
        numpy.ndarray: The autocorrelation function.
    """
    x = np.atleast_1d(x)
    mean, var = np.mean(x), np.var(x)
    xp = x - mean
    corr = np.correlate(xp, xp, mode='full')[len(x)-1:]
    return corr / (var * np.arange(len(x), 0, -1))

def autocorr_time(M, chain):
    """
    Compute the autocorrelation time of a chain up to lag M.

    Args:
        M (int): The maximum lag.
        chain (numpy.ndarray): The chain for which to compute the autocorrelation time.

    Returns:
        float: The autocorrelation time.
    """
    acors = autocorr_func_1d(chain)
    return 1 + 2 * np.sum(acors[:M])

def iterative_autocorr_time(chain):
    """
    Compute the autocorrelation time of a chain iteratively.

    Args:
        chain (numpy.ndarray): The chain for which to compute the autocorrelation time.

    Returns:
        float: The autocorrelation time.
    """
    M = 1
    while True:
        t_next = autocorr_time(M+1, chain)
        t_curr = autocorr_time(M, chain)
        if t_next > t_curr:
            return t_curr
        M += 1

if __name__ == "__main__":
    chain = np.random.randn(10000)  # Replace with your chain of samples
    Ms = np.arange(1, 1000)  # Replace with your range of M values

    taus = [autocorr_time(M, chain) for M in Ms]
    plt.plot(Ms, taus, label='Estimated $\\tau$')

    tau_iter = iterative_autocorr_time(chain)
    plt.axhline(tau_iter, color='r', linestyle='--', label='Iterative estimate')

    plt.xlabel('M')
    plt.ylabel('$\\tau$')
    plt.legend()
    plt.show()
