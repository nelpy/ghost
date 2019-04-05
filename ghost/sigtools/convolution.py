"""This file contains useful functions to perform convolutions"""

__all__ = ['fastconv_scipy', 'fastconv_fftw',
           'fastconv_freq_scipy', 'fastconv_freq_fftw']

import numpy as np
from scipy.fftpack import next_fast_len, fft, ifft
from scipy.signal import fftconvolve
from multiprocessing import cpu_count

from .fourier import clean_scipy_cache

try:
    import pyfftw
except:
    pass

def fastconv_scipy(signal, kernel, *, mode=None, fft_length=None):
    """Computes a convolution with scipy's fftpack, using the
    convolution theorem. This function is more memory efficient
    than computing the convolution on the entire data at once.
    
    Parameters
    ----------
    signal : np.ndarray, 1D
        Input signal
    kernel : np.ndarray, 1D
        The kernel with which to convolve the signal
    mode : 'full', 'same', or 'valid', optional
        The type of convolution to perform.
        Default is 'same'.
    fft_length : float, optional
        FFT length to use when computing the convolution.
        Default is 16384; if the kernel is larger than this,
        than the closest power of 4 greater than the kernel
        length is chosen.

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

    if fft_length is not None:
        if fft_length < M:
            raise ValueError("FFT length must be at least the kernel"
                             " size of {}".format(M))
    else:   # Choose good fft_length
        fft_length = 16384
        while fft_length < M:
            fft_length *= 4

    chunksize = fft_length - M + 1

    res = np.zeros(tot_length, dtype='<c16')
    starts = range(0, N, chunksize)
    for start in starts:
        length = min(chunksize, N - start)
        conv_fd = (fft(signal[start:start+length], n=fft_length) 
                   * fft(kernel, n=fft_length))
        conv_chunk = ifft(conv_fd)[..., :length + M - 1]
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

def fastconv_fftw(signal, kernel, *, mode=None, chunksize=None, 
                  fft_length=None):
    """Computes a convolution with FFTW, using the convolution
    theorem. This function has been written to minimize memory
    usage.
    
    Parameters
    ----------
    signal : np.ndarray, 1D
        Input signal
    kernel : np.ndarray, 1D
        The kernel with which to convolve the signal
    mode : 'full', 'same', or 'valid', optional
        The type of convolution to perform.
        Default is 'same'.
    fft_length : float, optional
        FFT length to use when computing the convolution.
        Default is 16384; if the kernel is larger than this,
        than the closest power of 4 greater than the kernel
        length is chosen.

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

    if fft_length is not None:
        if fft_length < M:
            raise ValueError("FFT length must be at least the kernel"
                             " size of {}".format(M))
    else:   # Choose good fft_length
        fft_length = 16384
        while fft_length < M:
            fft_length *= 4

    chunksize = fft_length - M + 1

    # The convention in many textbooks is to write the time-domain
    # signal as lower-case and the Fourier Transform as capital-case
    # so we will follow that here
    x = pyfftw.zeros_aligned(fft_length, dtype='complex128')
    X = pyfftw.zeros_aligned(fft_length, dtype='complex128')
    fft_sig = pyfftw.FFTW( x, X, 
                           direction='FFTW_FORWARD',
                           flags=['FFTW_ESTIMATE'], 
                           threads=cpu_count(), 
                           planning_timelimit=5 )

    y = pyfftw.zeros_aligned(fft_length, dtype='complex128')
    Y = pyfftw.zeros_aligned(fft_length, dtype='complex128')
    fft_kernel = pyfftw.FFTW( y, Y, 
                              direction='FFTW_FORWARD', 
                              flags=['FFTW_ESTIMATE'], 
                              threads=cpu_count(), 
                              planning_timelimit=5 )
    
    # Quick sanity checks that we're using FFTW correctly
    x[..., :chunksize] = signal[..., :chunksize]
    fft_sig(normalise_idft=True)
    assert np.allclose(X[..., :fft_length],
                       fft(signal[..., :chunksize], fft_length))

    # Check that we get the signal back when taking the IFFT of the FFT
    xfd = pyfftw.zeros_aligned(fft_length, dtype='complex128')
    xback = pyfftw.zeros_aligned(fft_length, dtype='complex128')
    fft_sig_backward = pyfftw.FFTW( xfd, xback, 
                           direction='FFTW_BACKWARD',
                           flags=['FFTW_ESTIMATE'], 
                           threads=cpu_count(), 
                           planning_timelimit=5 )
    xfd[...] = X
    fft_sig_backward(normalise_idft=True)
    assert np.allclose(xback[..., :chunksize], x[..., :chunksize])

    y[..., :M] = kernel
    # Notice that once we take the FFT of the kernel, we never have to
    # do it again! Just multiply with the FFT of each chunk. This saves
    # us computation
    fft_kernel()
    assert np.allclose(Y, fft(kernel, fft_length))

    # We don't care about the convolution looks like in the frequency
    # domain so we can compute an in-place FFT to save memory (and
    # potentially computation time)
    conv_fd = pyfftw.zeros_aligned(fft_length, dtype='complex128')
    conv_td = pyfftw.zeros_aligned(fft_length, dtype='complex128')
    fft_conv_inv = pyfftw.FFTW(conv_fd, conv_td,
                          direction='FFTW_BACKWARD', 
                          flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'], 
                          threads=cpu_count(), 
                          planning_timelimit=5 )

    res = np.zeros(tot_length, dtype='<c16')
    starts = range(0, N, chunksize)
    for start in starts:
        length = min(chunksize, N - start)
        # Ensure that the input data buffer for the FFT on iteration
        # i gets completely overwritten by the data on iteration i+1
        # (and any extra space in the buffer is filled with 0s)
        x[..., :length] = signal[..., start:start+length]
        x[..., length:] = 0
        fft_sig(normalise_idft=True)
        conv_fd[...] = X * Y
        fft_conv_inv(normalise_idft=True)
        res[..., start:start+length+M-1] += conv_td[..., :length+M-1]

    if mode == 'full':
        newsize = tot_length
    elif mode == 'same':
        newsize = N
    else:  # valid
        newsize = N - M + 1
    st_ind = (tot_length - newsize) // 2

    return res[..., st_ind:st_ind+newsize]

def fastconv_freq_scipy(signal_td, kernel_fd, kernel_len,
                        *, mode=None, chunksize=None):
    """Computes a convolution with scipy's fftpack, using
    the convolution theorem, where the signal is in the
    time domain but the kernel is in the frequency domain.
    This function is more memory efficient than computing
    the convolution on the entire data at once.

    Parameters
    ----------
    signal : np.ndarray, 1D
        Input signal, in the time domain
    kernel_fd : np.ndarray, 1D
        Kernel, in the frequency domain
    kernel_len : float
        The actual length of the kernel, since the frequency
        samples may be padded
    mode : 'full', 'same', or 'valid', optional
        The type of convolution to perform.
        Default is 'same'.
    chunksize : float, optional
        The chunksize to use for the signal when performing
        the convolution.
        Default is kernel_fd.size - kernel_len

    Returns
    -------
    A 1D array containing the result of the convolution
    """

    N = signal_td.shape[-1]
    M = kernel_len
    tot_length = N + M - 1

    if signal_td.ndim != 1:
        raise ValueError("Signal must be 1D")
    if kernel_fd.ndim != 1:
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
        chunksize = min(kernel_fd.size - M + 1, 30000)

    if chunksize + M - 1 > kernel_fd.size:
        raise ValueError("There must be at least {} frequency samples of the"
                         " kernel to compute the convolution but only got {}"
                         .format(chunksize + M - 1, kernel_fd.size))

    res = np.zeros(tot_length, dtype='<c16')
    starts = range(0, N, chunksize)
    for start in starts:
        # This part is critical to get right. We first take a DFT
        # of each chunk, with *total length* the same as that of the
        # frequency samples. Then multiply and excise the right
        # length for each chunk
        length = min(chunksize, N - start)
        signal_fd = fft(signal_td[start:start+length], n=kernel_fd.size)
        conv_fd = signal_fd * kernel_fd
        conv_td = ifft(conv_fd)[..., :length+M-1]
        res[start:start+len(conv_td)] += conv_td
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

def fastconv_freq_fftw(signal_td, kernel_fd, kernel_len,
                        *, mode=None, chunksize=None):
    """Computes a convolution with FFTW, using the convolution
    theorem, where the signal is in the time domain but the
    kernel is in the frequency domain. This function has been
    written to minimize memory usage.

    Parameters
    ----------
    signal : np.ndarray, 1D
        Input signal, in the time domain
    kernel_fd : np.ndarray, 1D
        Kernel, in the frequency domain
    kernel_len : float
        The actual length of the kernel, since the frequency
        samples may be padded
    mode : 'full', 'same', or 'valid', optional
        The type of convolution to perform.
        Default is 'same'.
    chunksize : float, optional
        The chunksize to use for the signal when performing
        the convolution.
        Default is kernel_fd.size - kernel_len

    Returns
    -------
    A 1D array containing the result of the convolution
    """

    N = signal_td.shape[-1]
    M = kernel_len
    tot_length = N + M - 1

    if signal_td.ndim != 1:
        raise ValueError("Signal must be 1D")
    if kernel_fd.ndim != 1:
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
        chunksize = min(kernel_fd.size - M + 1, 30000)

    if chunksize + M - 1 > kernel_fd.size:
        raise ValueError("There must be at least {} frequency samples of the"
                         " kernel to compute the convolution but only got {}"
                         .format(chunksize + M - 1, kernel_fd.size))

    x = pyfftw.zeros_aligned(kernel_fd.size, dtype='complex128')
    X = pyfftw.zeros_aligned(kernel_fd.size, dtype='complex128')
    fft_sig = pyfftw.FFTW( x, X,
                           direction='FFTW_FORWARD',
                           flags=['FFTW_ESTIMATE'],
                           threads=cpu_count(), 
                           planning_timelimit=5 )

    # We don't care about what the convolution looks like in the
    # frequency domain so we do an in-place FFT to save memory
    conv_fd = pyfftw.zeros_aligned(kernel_fd.size, dtype='complex128')
    conv_td = pyfftw.zeros_aligned(kernel_fd.size, dtype='complex128')
    fft_conv_inv = pyfftw.FFTW( conv_fd, conv_td,
                           direction='FFTW_BACKWARD',
                           flags=['FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'],
                           threads=cpu_count(),
                           planning_timelimit=5 )

    res = np.zeros(tot_length, dtype='<c16')
    starts = range(0, N, chunksize)
    for start in starts:
        # This part is critical to get right. We first take a DFT
        # of each chunk, with *total length* the same as that of the
        # frequency samples. Then multiply and excise the right
        # length for each chunk
        length = min(chunksize, N - start)

        # Ensure that we overwrite *all* of the input data buffer
        # for each iteration so that there's nothing remaining
        # from the last iteration
        x[..., :length] = signal_td[start:start+length]
        x[..., length:] = 0
        fft_sig(normalise_idft=True)

        # If the strides and alignment are right, the arrays associated
        # with a PyFFTW.FFTW object might actually share memory with the
        # array whose values are being accessed. This is a great way to
        # be memory efficient. But we also need to be especially careful
        # since we are using an in-place transform. If the line below
        # were simply conv_fd[...] = kernel_fd, kernel_fd might be
        # destroyed after taking the transform and that's no good.
        # However, in this case, the rvalue of the assignment operator
        # is a product, and we know a product creates a new array, so
        # we're safe and shouldn't have to worry about the rvalue arrays
        # getting destroyed
        conv_fd[...] = X * kernel_fd
        fft_conv_inv(normalise_idft=True)

        res[start:start+length+M-1] += conv_td[:length+M-1]

    if mode == 'full':
        newsize = tot_length
    elif mode == 'same':
        newsize = N
    else:  # valid
        newsize = N - M + 1
    st_ind = (tot_length - newsize) // 2
    
    return res[..., st_ind:st_ind+newsize]