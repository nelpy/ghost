import logging
import numpy as np
import time
import warnings

from abc import ABC
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from . import wavelet as wavedef
from . import morlet
from . import morse
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
    """
    def __init__(self, *, wavelet=None):

        if wavelet is None:
            wavelet = morse.Morse()
        self._wavelet = wavelet

        # I make sure to initialize all attributes in the
        # init. Makes it easier to know ahead of time in one
        # place what the expected attributes are
        self._freqs = None
        self._w = None
        self._fs = None

    def cwt(self, obj, fs, *, lengths=None, freq_limits=None, freqs=None,
            spectrogram=None, output=None, parallel=None, verbose=None):
        """Does a continuous wavelet transform. The output is a power
        spectrogram
        
        Parameters
        ----------
        obj : numpy.ndarray or nelpy.RegularlySampledAnalogSignalArray
            Must have only one non-singleton dimension
        fs : float
            Sampling rate of the data in Hz.
        lengths : array-like, optional
            Array-like object that specifies the blocksize (in elements)
            of the data that should be treated as a contiguous segment.
            If the input is a nelpy.RegularlySampledAnalogSignalArray,
            the lengths will be automatically inferred.
        freq_limits : list, optional
            List of [lower_bound, upper_bound] for frequencies to use,
            in units of Hz. Note that a reference set of frequencies
            is generated on the shortest segment of data since that
            determines the lowest frequency that can be used. If the
            bounds specified by 'freq_limits' are outside the bounds
            determined by the reference set, 'freq_limits' will be
            adjusted to be within the bounds of the reference set.
        freqs : array-like, optional
            Frequencies to analyze, in units of Hz.
            Note that a reference set of frequencies is computed on the
            shortest segment of data since that determines the lowest
            frequency that can be used. If any frequencies specified in
            'freqs' are outside the bounds determined by the reference
            set, 'freqs' will be adjusted such that all frequencies in
            'freqs' will be within those bounds.
        spectrogram : string, optional
            Type of spectrogram to generate, 'amplitude', or 'power'
            Default is 'amplitude'
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

        if spectrogram is None:
            spectrogram = 'amplitude'
        if spectrogram not in ('amplitude', 'power'):
            raise ValueError("Option 'spectrogram' must be 'amplitude' or"
                             " 'power' but got {}".format(spectrogram))

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

        w_ref = self.wavelet.generate_freqs(np.min(lengths))

        if freq_limits is not None and freqs is not None:
            raise ValueError("freq_limits and freqs cannot both be used at the"
                             " same time. Either specify one or the either, or"
                             " leave both as unspecified")

        if freq_limits is not None:
            freq_bounds = [freq_limits[0], freq_limits[-1]]
            lb, ub = self._check_freq_bounds(freq_bounds,
                            self._norm_radians_to_hz(w_ref))
            mask = np.logical_and(w_ref >= self._hz_to_norm_radians(lb),
                                  w_ref <= self._hz_to_norm_radians(ub))
            w = w_ref[mask]

        if freqs is not None:
            freq_bounds = [np.min(freqs), np.max(freqs)]
            lb, ub = self._check_freq_bounds(freq_bounds,
                            self._norm_radians_to_hz(w_ref))
            mask = np.logical_and(freqs >= lb, freqs <= ub)

        self._w = w
        self._freqs = self._norm_radians_to_hz(w)
        self._wavelet.fs = self._fs  # Set the wavelet's sampling rate accordingly

        # Set up array as C contiguous since we will be iterating row-by-row
        out_array = np.zeros((len(self._freqs), input_asarray.shape[-1]))

        def wavelet_conv(param):
            """The function that does the wavelet convolution"""

            idx, radian_freq = param
            # We need to copy because we might have multiple instances of this function
            # running in parallel. Each instance needs its own wavelet with which to
            # do the convolution
            wv = self._wavelet.copy()
            wv.radian_freq = radian_freq
            kernel, _ = wv.get_wavelet(60000)

            if verbose:
                print("Processing frequency {} Hz".
                        format(self._norm_radians_to_hz(radian_freq)))
            
            for ii in range(len(ei)-1): # process within epochs
                start, stop = ei[ii], ei[ii+1]
                res = convfun(input_asarray[..., start:stop], kernel)
                if spectrogram == 'amplitude':
                    out_array[idx, start:stop] = np.abs(res)
                else:
                    out_array[idx, start:stop] = np.square(np.abs(res))

        if parallel:
            warnings.warn(("You may not get as large a speedup as you hoped for by"
                           " running this function in parallel (single process,"
                           " multithreaded). Experimentation recommended."))
            pool = ThreadPool(cpu_count())
            it = [(idx, radian_freq) for idx, radian_freq in enumerate(self._w)]
            start_time = time.time()
            pool.map(wavelet_conv, it, chunksize=1)
            pool.close()
            pool.join()
        else:
            start_time = time.time()
            for ii, radian_freq in enumerate(self._w):
                wavelet_conv((ii, radian_freq))

        if verbose:
            print("Elapsed time (only wavelet convolution): {} seconds"
                  " to analyze {} frequencies".
                  format(time.time() - start_time, self._w.size))

        return post.output_numpy_or_asa(obj, 
                                        out_array.T,
                                        output_type=output, 
                                        labels=self._freqs)

    def _norm_radians_to_hz(self, val):

        return val / np.pi * self._fs / 2

    def _hz_to_norm_radians(self, val):

        return val / (self._fs / 2) * np.pi

    def _check_freq_bounds(self, freq_bounds, freqs_ref):

        # Inputs are Hz, outputs are Hz
        lb = np.min(freq_bounds)
        ub = np.max(freq_bounds)
        lb_ref = np.min(freqs_ref)
        ub_ref = np.max(freqs_ref)

        if lb < lb_ref:
            logging.warning("Specified lower bound was {:.3f} Hz but lower bound"
                            " computed on shortest segment was determined"
                            " to be {:.3f} Hz. The lower bound will be adjusted"
                            " upward to {:.3f} Hz accordingly".
                            format(lb, lb_ref, lb_ref))
            lb = lb_ref
        if ub > ub_ref:
            logging.warning("Specified upper bound was {:.3f} Hz but upper bound"
                            " was determined to be {:.3f} Hz. The upper bound"
                            " will be adjusted downward to {:.3f} Hz accordingly".
                            format(ub, ub_ref, ub_ref))
            ub = ub_ref

        return lb, ub

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
    def freqs(self, val):

        raise ValueError("Setting frequencies outside of cwt()"
                         " is disallowed. Please use the cwt()"
                         " interface if you want to use a"
                         " different set of frequencies for"
                         " the cwt")

    @property
    def radian_freqs(self):

        return self._hz_to_norm_radians(self._freqs)

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
