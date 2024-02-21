import numpy as np
import matplotlib.pyplot as plt

def autocorr_func_1d(x, norm=True):
    """
    Compute the 1-dimensional autocorrelation function of a signal.

    Args:
        x (numpy.ndarray): The input signal.
        norm (bool, optional): If True, normalize the autocorrelation function. Default is True.

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
    Estimate the autocorrelation time τ for a given window size M.

    Args:
        M (int): The window size.
        chain (numpy.ndarray): The chain of samples.

    Returns:
        float: The estimated autocorrelation time τ.
    """
    acors = autocorr_func_1d(chain)
    return 1 + 2*np.sum(acors[:M])

def iterative_autocorr_time(chain):
    """
    Perform an iterative procedure to estimate the autocorrelation time τ.

    Args:
        chain (numpy.ndarray): The chain of samples.

    Returns:
        float: The estimated autocorrelation time τ.
    """
    M = 1
    while True:
        t_next = autocorr_time(M+1, chain)
        t_curr = autocorr_time(M, chain)
        if t_next > t_curr:
            return t_curr
        M += 1

chain = np.random.randn(10000)  # replace with your chain of samples
Ms = np.arange(1, 1000)  # replace with your range of M values

# Compute and plot estimates for different segments
num_segments = 10
segment_length = len(chain) // num_segments
for i in range(num_segments):
    segment = chain[i*segment_length:(i+1)*segment_length]
    taus_segment = [autocorr_time(M, segment) for M in Ms]
    plt.plot(Ms, taus_segment, color='black')

# Compute and plot estimate for full chain
taus_full = [autocorr_time(M, chain) for M in Ms]
plt.plot(Ms, taus_full, color='orange', label='Full chain')

# Compute and plot iterative estimate
tau_iter = iterative_autocorr_time(chain)
plt.axhline(tau_iter, color='r', linestyle='--', label='Iterative estimate')

plt.xlabel('M')
plt.ylabel('$\\tau$')
plt.title(f'Iterative estimate: {tau_iter:.2f}')
plt.legend()
plt.show()
