"""Support for different Fourier and inverse Fourier transforms"""

import logging
import numpy as np
from scipy.fftpack import fft as sff
from . import convolution

__all__ = ['clean_scipy_cache', 'chirpz_dft']

def clean_scipy_cache():
    sff.destroy_zfft_cache()
    sff.destroy_zfftnd_cache()
    sff.destroy_drfft_cache()
    sff.destroy_cfft_cache()
    sff.destroy_cfftnd_cache()
    sff.destroy_rfft_cache()
    sff.destroy_ddct2_cache()
    sff.destroy_ddct1_cache()
    sff.destroy_dct2_cache()
    sff.destroy_dct1_cache()
    sff.destroy_ddst2_cache()
    sff.destroy_ddst1_cache()
    sff.destroy_dst2_cache()
    sff.destroy_dst1_cache()

def chirpz_dft(x):
    """Computes the DFT of a signal using the Chirp Z-transform.
    This function is in general slower than FFTW but can be orders
    of magnitude faster than scipy's fft for prime-length data.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input data. Must be 1D

    Returns
    -------
    The DFT of x

    """

    if x.ndim != 1:
        raise ValueError("Data must be 1-dimensional")

    try:
        import pyfftw
        convfun = convolution.fastconv_fftw
    except:
        logging.warning("Module 'pyfftw' not found, using scipy backend")
        convfun = convolution.fastconv_scipy

    N = x.shape[-1]
    n = np.arange(0, N) # Sequence length

    W = np.exp(-1j*np.pi*n*n/N) # Chirp signal

    x_chirp = x*W   # Modulate with chirp

    kernel = np.hstack( (W[N:0:-1].conj(), W.conj()) )

    # In the convolution, can choose an efficient length
    # DFT - OK to pad input data
    y = convfun(x_chirp, kernel, mode='full')

    return y[N-1:N-1+N] * W


