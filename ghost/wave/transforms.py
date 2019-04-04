import logging
import sys
import numpy as np
import time
import warnings

from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from . import wavelet as wav
from . import morlet
from .. import sigtools
from .. import preprocessing as pre
from .. import postprocessing as post

try:
    import nelpy as nel
except:
    pass

__all__ = ['ContinuousWaveletTransform']

class WaveletTransform(ABC):

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

class ContinuousWaveletTransform(WaveletTransform):
    """This class is used to perform continuous wavelet transforms"""

    def __init__(self, *, freqs=None):

        if freqs is None:
            # User should really pass in the frequencies they want to analyze
            # since it's usually a band they are interested in. Doing the
            # wavelet transform outside this band is wasteful
            freqs = np.arange(5, 401, 5)   # cover LFP spectrum

        self.freqs = freqs

    def cwt(self, obj, fs, *, lengths=None, output=None,
            wavelet=None, spectrogram=None, parallel=None, 
            backend=None, verbose=None):

        """Does a continuous wavelet transform
        
        Parameters
        ----------
        obj : numpy.ndarray or nelpy.RegularlySampledAnalogSignalArray
            Must have only one dimension after being squeezed
        lengths : array-like
            Array-like object that specifies the blocksize (in elements) of
            the data that should be treated as a contiguous segment
        output : string, optional
            Specify output='asa' to return a nelpy ASA
        wavelet : A wavelet that inherits from ghost.Wavelet, optional
        spectrogram : string, optional
            Type of spectrogram to return, 'power', or 'amplitude'
            Default is 'power'
        parallel : boolean, optional
            Whether to run this function in parallel or not, using a
            single process, multithreaded model.
            Default is False.
        backend : string, optional
            FFT backend to use. 'scipy' or 'fftw'
            Default is 'scipy', which uses scipy's fftpack
        verbose : boolean, optional
            Whether to print messages displaying this function's progress.
            Default is False.
        """

        if output is not None:
            if not output in ('asa'):
                raise TypeError("Output type {} not supported".format(output))

        if wavelet is None:
            wavelet = morlet.Morlet(fs=fs)
        if wavelet.fs != fs:
            raise ValueError("Wavelet must have same sampling rate as input data")
        if not isinstance(wavelet, wav.Wavelet):
            raise TypeError("Wavelet must inherit from the wavelet object" 
                            " defined in wavelet.py")

        if spectrogram is None:
            spectrogram = 'power'
        if not spectrogram in ('power', 'amplitude'):
            raise ValueError("Spectrogram type '{}' is not supported".format(spectrogram))

        if parallel is None:
            parallel = False
        if parallel not in (True, False):
            raise ValueError("'parallel' must be either True or False")

        if backend is None:
            backend = 'scipy'
        if not backend in ('scipy', 'fftw'):
            raise ValueError("FFT backend '{}' is not valid".format(backend))
        if backend == 'scipy':
            convfun = sigtools.fastconv_scipy
        if backend == 'fftw':
            convfun = sigtools.fastconv_fftw

        if verbose is None:
            verbose = False
        if verbose not in (True, False):
            raise ValueError("'verbose' must be either True or False")

        input_asarray, lengths = pre.standardize_1d_asa(obj, lengths=lengths)
        input_asarray -= np.mean(input_asarray)
        ei = np.insert(np.cumsum(lengths), 0, 0) # epoch indices
        # Set up array as C contiguous since we will be iterating row-by-row
        out_array = np.zeros((len(self.freqs), input_asarray.shape[-1]))

        def wavelet_conv(param):
            """The function that does the wavelet convolution"""

            idx, freq = param
            # We need to copy because we might have multiple instances of this function
            # running in parallel. Each instance needs its own wavelet with which to
            # do the convolution
            wv = wavelet.copy()
            wv.freq = freq
            kernel = wv.get_wavelet()
            if kernel.ndim != 1:
                raise ValueError("Wavelet must be 1D")

            if verbose:
                print("Processing frequency {} Hz".format(freq))
            
            for ii in range(len(ei)-1): # process within epochs
                start, stop = ei[ii], ei[ii+1]
                res = convfun(input_asarray[..., start:stop], kernel)
                if spectrogram == 'power':
                    out_array[idx, start:stop] = np.square(np.abs(res))
                else: # output must be 'amplitude'
                    out_array[idx, start:stop] = np.abs(res)

        if parallel:
            warnings.warn(("You may not get as large a speedup as you hoped for by"
                           " running this function in parallel (single process,"
                           " multithreaded). Experimentation recommended."))
            pool = ThreadPool(cpu_count())
            it = [(idx, freq) for idx, freq in enumerate(self.freqs)]
            start_time = time.time()
            pool.map(wavelet_conv, it, chunksize=1)
            pool.close()
            pool.join()
        else:
            start_time = time.time()
            for ii, freq in enumerate(self.freqs):
                wavelet_conv((ii, freq))

        if verbose:
            print("Elapsed time (only wavelet convolution): {} seconds".format(time.time() - start_time))

        return post.output_numpy_or_asa(obj, 
                                        out_array.T,
                                        output_type=output, 
                                        labels=self.freqs)
