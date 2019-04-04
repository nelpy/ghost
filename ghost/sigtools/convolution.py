"""This file contains useful functions to perform convolutions"""

__all__ = ['fastconv_scipy', 'fastconv_fftw']

import numpy as np
import scipy.fftpack._fftpack as sff
from scipy.fftpack import next_fast_len, fft, ifft
from scipy.signal import fftconvolve
from multiprocessing import cpu_count

try:
    import pyfftw
except:
    pass

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

def fastconv_scipy(signal, kernel, *, mode=None, chunksize=None):
    """Computes a convolution using the convolution theorem with
    scipy's fftpack. This function tries to be as memory efficient
    as possible by processing the data in chunks.
    
    Parameters
    ----------
    signal : np.ndarray, 1D
        Input signal
    kernel : np.ndarray, 1D
        The kernel to convolve the signal with
    mode : 'full', 'same', or 'valid', optional
        The type of convolution to perform.
        Default is 'same'.
    chunksize : float, optional
        The chunksize to use for the signal when performing
        the convolution.
        Default is 30000.

    Returns
    -------
    A 1D array containing the result of the convolution
    """

    N = signal.shape[-1]
    M = kernel.shape[-1]
    tot_length = N + M - 1

    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    if kernel.ndim != 1:
        raise ValueError("Kernel must be 1D")

    if mode is None:
        mode = 'same'
    if mode not in ('full', 'same', 'valid'):
        raise ValueError("Mode must be 'full', 'same', or 'valid'")
    if mode == 'valid':
        if N < M:
            raise ValueError("Cannot do a 'valid' convolution because "
                             "the input is shorter than the kernel")

    if chunksize is None:
        chunksize = np.minimum(N, 30000)

    res = np.zeros(tot_length, dtype='<c16')
    starts = range(0, N, chunksize)
    for start in starts:
        conv_chunk = fftconvolve(signal[start:start+chunksize], kernel)
        res[start:start+len(conv_chunk)] += conv_chunk
        # Usually the cache is to blame for memory errors
        clean_scipy_cache()

    if mode == 'full':
        newsize = tot_length
    elif mode == 'same':
        newsize = N
    else:  # valid
        newsize = N - M + 1
    st_ind = (tot_length - newsize) // 2
    
    return res[..., st_ind:st_ind+newsize]

def fastconv_fftw(signal, kernel, *, mode=None, chunksize=None):
    """Computes a convolution using the convolution theorem using
    FFTW. This function tries to be as memory efficient as possible
    by processing the data in chunks and minimizing array copying.
    
    Parameters
    ----------
    signal : np.ndarray, 1D
        Input signal
    kernel : np.ndarray, 1D
        The kernel to convolve the signal with
    mode : 'full', 'same', or 'valid', optional
        The type of convolution to perform.
        Default is 'same'.
    chunksize : float, optional
        The chunksize to use for the signal when performing
        the convolution.
        Default is 30000.

    Returns
    -------
    A 1D array containing the result of the convolution
    """

    N = signal.shape[-1]
    M = kernel.shape[-1]
    tot_length = N + M - 1

    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    if kernel.ndim != 1:
        raise ValueError("Kernel must be 1D")

    if mode is None:
        mode = 'same'
    if mode not in ('full', 'same', 'valid'):
        raise ValueError("Mode must be 'full', 'same', or 'valid'")
    if mode == 'valid':
        if N < M:
            raise ValueError("Cannot do a 'valid' convolution because "
                             "the input is shorter than the kernel")

    if chunksize is None:
        chunksize = min(len(signal), 30000)

    fast_chunk_len = pyfftw.next_fast_len(chunksize + M - 1)

    # The convention in many textbooks is to write the time-domain
    # signal as lower-case and the Fourier Transform as capital-case
    # so we will follow that here
    x = pyfftw.zeros_aligned(fast_chunk_len, dtype='complex128')
    X = pyfftw.zeros_aligned(fast_chunk_len, dtype='complex128')
    fft_sig = pyfftw.FFTW( x, X, 
                           direction='FFTW_FORWARD',
                           flags=['FFTW_ESTIMATE'], 
                           threads=cpu_count(), 
                           planning_timelimit=5 )

    y = pyfftw.zeros_aligned(fast_chunk_len, dtype='complex128')
    Y = pyfftw.zeros_aligned(fast_chunk_len, dtype='complex128')
    fft_kernel = pyfftw.FFTW( y, Y, 
                              direction='FFTW_FORWARD', 
                              flags=['FFTW_ESTIMATE'], 
                              threads=cpu_count(), 
                              planning_timelimit=5 )
    
    # Quick sanity checks that we're using FFTW correctly
    x[..., :chunksize] = signal[0:0+chunksize]
    fft_sig(normalise_idft=True)
    assert np.allclose(X[:fast_chunk_len], fft(signal[0:0+chunksize], fast_chunk_len))

    # Check that we get the signal back when taking the IFFT of the FFT
    xfd = pyfftw.zeros_aligned(fast_chunk_len, dtype='complex128')
    xback = pyfftw.zeros_aligned(fast_chunk_len, dtype='complex128')
    fft_sig_backward = pyfftw.FFTW( xfd, xback, 
                           direction='FFTW_BACKWARD',
                           flags=['FFTW_ESTIMATE'], 
                           threads=cpu_count(), 
                           planning_timelimit=5 )
    xfd[...] = X
    fft_sig_backward(normalise_idft=True)
    assert np.allclose(xback[:chunksize], x[:chunksize])

    y[..., :M] = kernel
    # Notice that once we take the FFT of the kernel, we never have to
    # do it again! Just multiply with the FFT of each chunk. This saves
    # us computation
    fft_kernel()
    assert np.allclose(Y, fft(kernel, fast_chunk_len))

    conv_fd = pyfftw.zeros_aligned(fast_chunk_len, dtype='complex128')
    conv_td = pyfftw.zeros_aligned(fast_chunk_len, dtype='complex128')
    # We don't care about saving the result of the convolution in the
    # frequency domain so we do an in-place FFT here to save memory
    fft_conv_inv = pyfftw.FFTW(conv_fd, conv_td, 
                          direction='FFTW_BACKWARD', 
                          flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'], 
                          threads=cpu_count(), 
                          planning_timelimit=5 )

    res = np.zeros(tot_length, dtype='<c16')
    starts = range(0, N, chunksize)
    for start in starts:
        length = min(chunksize, N - start)
        x[..., :length] = signal[..., start:start+length]
        fft_sig(normalise_idft=True)
        conv_fd[...] = Y * X
        fft_conv_inv(normalise_idft=True)
        res[..., start:start+length+M-1] += conv_td[..., :length+M-1]
        # If variable chunksizes are used, we should do:
        # (1) x[..., :length] = signal[..., start:start+chunksize]
        # (2) x[..., length:] = 0
        # This ensures the input data buffer for the FFT on iteration
        # i gets completely overwritten by the data on iteration i+1
        # (and any extra space in the buffer is filled with 0).
        # Variable length chunksizes only occur on the very last iteration
        # and so we don't need to bother with (2) because we aren't
        # taking the FFT another time. Eliminating the unecessary operation
        # thus saves us some computation.

    if mode == 'full':
        newsize = tot_length
    elif mode == 'same':
        newsize = N
    else:  # valid
        newsize = N - M + 1
    st_ind = (tot_length - newsize) // 2
    
    return res[..., st_ind:st_ind+newsize]