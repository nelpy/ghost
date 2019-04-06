import logging
import sys
import numpy as np
import time
import warnings

from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from . import wavelet as wavedef
from . import morlet
from .. import sigtools
from ..formats import preprocessing as pre
from ..formats import postprocessing as post

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
    """This class is used to perform continuous wavelet transforms

    Parameters
    ----------
    wavelet : of type ghost.Wavelet, optional
        The wavelet to use for this transform
    rectify : boolean, optional
        Whether to apply the correction in Liu 2007 i.e. divide the
        wavelet power by the scale to have a consistent definition
        of energy.
        Default is True.
    """
    def __init__(self, *, wavelet=None, rectify=None):

        if wavelet is None:
            wavelet = morlet.Morlet()
        self._wavelet = wavelet

        if rectify is None:
            rectify = True
        if rectify not in (True, False):
            raise ValueError("'rectify' must be either True or False")
        self._rectify = rectify

        # I make sure to initialize all attributes in the
        # init. Makes it easier to know ahead of time in one
        # place what the expected attributes are
        self._freqs = None
        self._fs = None

    def cwt(self, obj, fs, *, lengths=None, freqs=None, output=None,
            parallel=None, verbose=None):
        """Does a continuous wavelet transform. The output is a power
        spectrogram
        
        Parameters
        ----------
        obj : numpy.ndarray or nelpy.RegularlySampledAnalogSignalArray
            Must have only one dimension after being squeezed
        fs : float
            Sampling rate of the data in Hz.
        lengths : array-like
            Array-like object that specifies the blocksize (in elements) of
            the data that should be treated as a contiguous segment
        freqs : array-like, optional
            Frequencies to analyze. Units of Hz.
            Default is 5 to 400 Hz in steps of 5 Hz
        output : string, optional
            Specify output='asa' to return a nelpy ASA
        parallel : boolean, optional
            Whether to run this function in parallel or not, using a
            single process, multithreaded model.
            Default is False.
        verbose : boolean, optional
            Whether to print messages displaying this function's progress.
            Default is False.
        """

        self.fs = fs  # does input validation, nice!
        if freqs is None:
            freqs = np.arange(5, 401, 5)
        self.freqs = freqs  # also does input validation!

        if output is not None:
            if not output in ('asa'):
                raise TypeError("Output type {} not supported".format(output))

        if parallel is None:
            parallel = False
        if parallel not in (True, False):
            raise ValueError("'parallel' must be either True or False")

        if verbose is None:
            verbose = False
        if verbose not in (True, False):
            raise ValueError("'verbose' must be either True or False")

        try:
            import pyfftw
            convfun = sigtools.fastconv_fftw
        except:
            warnings.warn("Module 'pyfftw' not found, using scipy backend")
            convfun = sigtools.fastconv_scipy

        input_asarray, lengths = pre.standard_format_1d_asa(obj, lengths=lengths)
        input_asarray -= np.mean(input_asarray)
        ei = np.insert(np.cumsum(lengths), 0, 0) # epoch indices
        # Set up array as C contiguous since we will be iterating row-by-row
        out_array = np.zeros((len(self._freqs), input_asarray.shape[-1]))

        self._wavelet.fs = self._fs  # Set the wavelet's sampling rate accordingly

        def wavelet_conv(param):
            """The function that does the wavelet convolution"""

            idx, freq = param
            # We need to copy because we might have multiple instances of this function
            # running in parallel. Each instance needs its own wavelet with which to
            # do the convolution
            wv = self._wavelet.copy()
            wv.freq = freq
            kernel = wv.get_wavelet()

            if verbose:
                print("Processing frequency {} Hz".format(freq))
            
            for ii in range(len(ei)-1): # process within epochs
                start, stop = ei[ii], ei[ii+1]
                res = convfun(input_asarray[..., start:stop], kernel)
                out_array[idx, start:stop] = np.square(np.abs(res))
                if self._rectify:
                    out_array[idx, start:stop] /= wv.scale


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
            for ii, freq in enumerate(self._freqs):
                wavelet_conv((ii, freq))

        if verbose:
            print("Elapsed time (only wavelet convolution): {} seconds".format(time.time() - start_time))

        return post.output_numpy_or_asa(obj, 
                                        out_array.T,
                                        output_type=output, 
                                        labels=self._freqs)

    @property
    def fs(self):

        return self._fs

    @fs.setter
    def fs(self, samplerate):

        if samplerate <= 0:
            raise ValueError("Sampling rate must be positive")
        
        self._fs = samplerate

    @property
    def freqs(self):
        """The frequencies this transform analyzes, in Hz"""

        return self._freqs

    @freqs.setter
    def freqs(self, frequencies):
        
        if np.any(frequencies) < 0:
            logging.warning("Negative frequencies will cause undefined "
                            " behavior. Use at your own risk")

        self._freqs = frequencies

    @property
    def wavelet(self):
        """Returns wavelet associated with this transform object"""

        return self._wavelet

    @wavelet.setter
    def wavelet(self, wav):
        
        if wav.fs != self._fs:
            raise ValueError("Wavelet must have same sampling rate as input data")
        if not isinstance(wav, wavedef.Wavelet):
            raise TypeError("The wavelet must be of type ghost.Wavelet")
        if wav.get_wavelet().ndim != 1:
            raise ValueError("Wavelet must be 1D")

        self._wavelet = wav
