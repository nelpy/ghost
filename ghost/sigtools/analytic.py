"""This file contains functions for computing the analytic signal"""

import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len
from multiprocessing import cpu_count

try:
    import pyfftw
except:
    pass

__all__ = ['analytic_signal_scipy', 'analytic_signal_fftw']

def analytic_signal_scipy(*args):
    # forward along to scipy's hilbert for now but later
    # should make this version's interface and functionality
    # similar to analytic_signal_fftw i.e. automatic padding
    return hilbert(args)


def analytic_signal_fftw(signal):
    """Computes the analytic signal x_a = x + iy where
    x is the original signal, and y is the Hilbert transform
    of x

    Parameters
    ----------
    signal : numpy.ndarray
        Must be 1D and real

    Returns
    -------
    A 1D array containing the analytic signal
    """
    
    if np.iscomplexobj(signal):
        raise ValueError("The input data must be real")
    
    # use size because empty arrays can still have non-zero shapes
    if signal.size == 0: 
        raise ValueError("Cannot compute analytic signal on an empty array")

    if signal.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
        
    L = signal.shape[-1]
    fast_len = pyfftw.next_fast_len(L)
    M = fast_len // 2 + 1

    # Generating the analytic signal involves taking the FT of the signal
    # and zeroing out the negative frequencies. Therefore, we don't
    # actually need to compute the negative frequencies. Instead, we
    # do a real-valued FFT to get the positive frequencies only. Note that
    # the signal must be real to use this strategy correctly!
    xtd = pyfftw.zeros_aligned(fast_len, dtype='float64')
    xfd = pyfftw.zeros_aligned(M, dtype='complex128')
    fft_sig = pyfftw.FFTW( xtd, xfd,
                           direction='FFTW_FORWARD',
                           flags=['FFTW_ESTIMATE'], 
                           threads=cpu_count(),
                           planning_timelimit=5 )
    
    # We don't care about the spectrum of the analytic signal
    # so we can compute an in-place FFT to save memory
    afd = pyfftw.zeros_aligned(fast_len, dtype='complex128')
    atd = pyfftw.zeros_aligned(fast_len, dtype='complex128')
    fft_analytic = pyfftw.FFTW( afd, atd,
                           direction='FFTW_BACKWARD',
                           flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'], 
                           threads=cpu_count(),
                           planning_timelimit=5 )

    xtd[..., :L] = signal
    fft_sig(normalise_idft=True)

    afd[0] = xfd[0] # This is true regardless if the DFT is odd or even length
    if fast_len & 1:  # odd
        afd[1:(fast_len + 1) // 2] = 2*xfd[1:]
    else:
        # The last value in an even-length real-valued DFT is both
        # positive and negative so we make sure to keep it in addition
        # to DC
        afd[fast_len // 2] = xfd[-1]
        afd[1:fast_len // 2] = 2*xfd[1:-1]
        
    fft_analytic(normalise_idft=True)

    return atd[..., :L]