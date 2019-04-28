import ghost.sigtools
import numpy as np
from scipy.signal import convolve
from scipy.fftpack import fft, ifft

def test_fastconv_time_domain():

    N = 10000
    M = 1000

    x = np.random.rand(N)
    y = np.random.rand(M)

    for mode in ('full', 'same', 'valid'):
        conv = convolve(x, y, mode=mode)
        conv_fftw = ghost.sigtools.fastconv_fftw(x, y, mode=mode, fft_length=2048)
        conv_scipy = ghost.sigtools.fastconv_scipy(x, y, mode=mode, fft_length=2048)

        assert np.allclose(conv_scipy, conv)
        assert np.allclose(conv_fftw, conv)
        assert np.allclose(conv_fftw, conv_scipy)

def test_fastconv_freq_domain():

    N = 10000
    M = 1000

    x = np.random.rand(N)
    y = np.random.rand(M)
    Y = fft(y, n=3000)

    for mode in ('full', 'same', 'valid'):

        conv = convolve(x, y, mode=mode)
        conv_scipy = ghost.sigtools.fastconv_freq_scipy(x, Y, 
                        len(y), mode=mode)
        conv_fftw = ghost.sigtools.fastconv_freq_fftw(x, Y, 
                        len(y), mode=mode)

        assert np.allclose(conv_scipy, conv)
        assert np.allclose(conv_fftw, conv)
        assert np.allclose(conv_fftw, conv_scipy)