import logging
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt

from abc import ABC
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from . import wavelet as wavedef
from . import morlet
from . import morse
from .. import sigtools
#from ..formats import standardize_asa_1d
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
        self._frequencies = None
        self._fs = None
        self._amplitude = None
        self._power = None
        self._time = None
        self._epochs = None

    @pre.standardize_asa(x='data', fs='fs', n_signals=1, 
                         class_method=True, abscissa_vals='timestamps')
    def transform(self, data, *, timestamps=None, fs=None, freq_limits=None,
                  freqs=None, voices_per_octave=None, parallel=None,
                  verbose=None, **kwargs):
        """Does a continuous wavelet transform. The output is a power
        spectrogram
        
        Parameters
        ----------
        data : numpy.ndarray or nelpy.RegularlySampledAnalogSignalArray
            Must have only one non-singleton dimension
        fs : float
            Sampling rate of the data in Hz.
        timestamps : np.ndarray, optional
            Timestamps corresponding to the data, in seconds.
            If None, they will be computed automatically based on the
            assumption that all the data are one contiguous block.
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
        voices_per_octave : float, optional
            Number of wavelet frequencies per octave
        parallel : boolean, optional
            Whether to run this function in parallel or not, using a
            single process, multithreaded model.
            Default is False.
        verbose : boolean, optional
            Whether to print messages displaying this function's progress.
            Default is False.
        """

        self.fs = fs  # does input validation, nice!
        self._time = timestamps # already sanitized by decorator

        if voices_per_octave is None:
            voices_per_octave = 16

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

        # need to copy data since we're mutating it
        input_asarray = data.squeeze().copy()
        input_asarray -= np.mean(input_asarray)
        epoch_bounds = kwargs.pop('epoch_bounds', None)
        if epoch_bounds is None:
            raise ValueError("decorator didn't do its job!")

        lengths = (np.diff(epoch_bounds, axis=1) - 1).astype(int)

        f_ref = self._norm_radians_to_hz(
                    self.wavelet.generate_omegas(np.min(lengths)))

        if freqs is not None and freq_limits is not None:
            raise ValueError("freq_limits and freqs cannot both be used at the"
                             " same time. Either specify one or the either, or"
                             " leave both as unspecified")

        if freqs is not None or freq_limits is not None:
            if freqs is not None:
                # just in case user didn't pass in sorted
                # frequencies after all
                freqs = np.sort(freqs)
                freq_bounds = [freqs[0], freqs[-1]]
            if freq_limits is not None:
                # just in case user didn't pass in limits as
                # [lower_bound, upper_bound]
                freq_limits = np.sort(freq_limits)
                freq_bounds = [freq_limits[0], freq_limits[-1]]

            lb, ub = self._check_freq_bounds(freq_bounds, f_ref)
            mask = np.logical_and(f_ref >= lb, f_ref <= ub)
            f = f_ref[mask]

        assert f[0] < f[-1]

        n_octaves = np.log2(f[-1] / f[0])
        J = np.floor(n_octaves) * voices_per_octave
        j = np.arange(J+1)
        f = f[0]*2**(j/voices_per_octave)
        self._frequencies = f
        self._wavelet.fs = self._fs  # Set the wavelet's sampling rate accordingly

        wave_lengths = self.wavelet.compute_lengths(
                            self._hz_to_norm_radians(self._frequencies)).tolist()

        # Set up array as C contiguous since we will be iterating row-by-row
        out_array = np.zeros((len(self._frequencies), input_asarray.shape[-1]))

        def wavelet_conv(param):
            """The function that does the wavelet convolution"""

            idx, freq, length = param
            # We need to copy because we might have multiple instances of this function
            # running in parallel. Each instance needs its own wavelet with which to
            # do the convolution
            wv = self._wavelet.copy()
            wv.norm_radian_freq = self._hz_to_norm_radians(freq)

            kernel, _ = wv(length)

            if verbose:
                print("Processing frequency {} Hz".format(freq))

            for start, stop in epoch_bounds: # process within epochs
                res = convfun(input_asarray[..., start:stop], kernel)
                out_array[idx, start:stop] = np.abs(res)

        if parallel:
            warnings.warn(("You may not get as large a speedup as you hoped for by"
                           " running this function in parallel (single process,"
                           " multithreaded). Experimentation recommended."))
            pool = ThreadPool(cpu_count())
            it = [(idx, freq, length) for idx, (freq, length) in 
                    enumerate(zip(np.flip(self._frequencies), np.flip(wave_lengths)))]
            start_time = time.time()
            pool.map(wavelet_conv, it, chunksize=1)
            pool.close()
            pool.join()
        else:
            start_time = time.time()
            # so that matrix is goes from high frequencies
            # at the origin to low frequencies
            for ii, (freq, length) in enumerate(
                zip(np.flip(self._frequencies), np.flip(wave_lengths))):
                wavelet_conv((ii, freq, length))

        if verbose:
            print("Elapsed time (only wavelet convolution): {} seconds"
                  " to analyze {} frequencies".
                  format(time.time() - start_time, self._frequencies.size))

        self._amplitude = out_array

    def plot(self, kind=None, timescale=None, logscale=None, 
             time_limits=None, freq_limits=None, ax=None, **kwargs):

        if kind is None:
            kind = 'amplitude'
        if kind not in ('amplitude, power'):
            raise ValueError("'kind' must be 'amplitude' or 'power', but"
                             " got {}".format(kind))

        if timescale is None:
            timescale = 'seconds'
            xlabel = "Time (sec)"
        if timescale not in ('seconds', 'minutes', 'hours'):
            raise ValueError("timescale must be 'seconds', 'minutes', or"
                            " 'hours' but got {}".format(timescale))

        if logscale is None:
            logscale = False
        if logscale not in (True, False):
            raise ValueError("'logscale' must be True or False but got {}".
                             format(logscale))

        if time_limits is None:
            time_slice = slice(None)
        else:
            try:
                # nelpy installed
                if isinstance(time_limits, nel.EpochArray):
                    time_limits = time_limits.data
                    if time_limits.shape[0] != 1:
                        raise ValueError("Detected {} epochs but can only restrict"
                                         " spectrogram plot to 1 epoch".
                                         format(time_limits.shape[0]))
                elif isinstance(time_limits, (np.ndarray, list)):
                    time_limits = np.array(time_limits)
                else:
                    raise TypeError("'time_limits' must be of type nelpy.EpochArray"
                                    "or np.ndarray but got {}".format(type(time_limits)))
            except NameError:
                # nelpy not installed
                if isinstance(time_limits, (np.ndarray, list)):
                    time_limits = np.array(time_limits)
                else:
                    raise TypeError("'time_limits' must be of type nelpy.EpochArray"
                                    "or np.ndarray but got {}".format(type(time_limits)))

            time_slice = self._restrict_plot_time(time_limits)

        if freq_limits is None:
            freq_slice = slice(None)
            data_freq_slice = slice(None)
        else:
            freq_slice, data_freq_slice = self._restrict_plot_freq(freq_limits)

        if kind == 'amplitude':
            data = self._amplitude
            title = 'Wavelet Amplitude Spectrogram'
        else:
            data = np.square(self._amplitude)
            title = 'Wavelet Power Spectrogram'

        timevec = self._time[time_slice]
        freqvec = self._frequencies[freq_slice]
        data = data[data_freq_slice, time_slice]

        if timescale == 'seconds':
            xlabel = "Time (sec)"
        elif timescale == 'minutes':
            timevec = timevec / 60
            xlabel = "Time (min)"
        else:
            timevec = timevec / 3600
            xlabel = "Time (hr)"

        if ax is None:
            ax = plt.gca()

        time, freq = np.meshgrid(timevec, np.flip(freqvec))
        ax.contourf(time, freq, data, **kwargs)
        if logscale:
            ax.set_yscale('log')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency (Hz)")

    def _norm_radians_to_hz(self, val):

        return val / np.pi * self._fs / 2

    def _hz_to_norm_radians(self, val):

        return val / (self._fs / 2) * np.pi

    def _check_freq_bounds(self, freq_bounds, freqs_ref):

        # Inputs are Hz, outputs are Hz
        lb = freq_bounds[0]
        ub = freq_bounds[-1]
        lb_ref = freqs_ref[0]
        ub_ref = freqs_ref[-1]

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

    def _restrict_plot_time(self, limits):

        tstart, tstop = np.searchsorted(self._time, limits)

        return slice(tstart, tstop)

    def _restrict_plot_freq(self, limits):

        f0, f1 = np.searchsorted(self._frequencies, limits)
        fstart = len(self._frequencies) - f1
        fstop = len(self._frequencies) - f0

        return slice(f0, f1), slice(fstart, fstop)

    @property
    def fs(self):

        return self._fs

    @fs.setter
    def fs(self, samplerate):

        if samplerate <= 0:
            raise ValueError("Sampling rate must be positive")
        
        self._fs = samplerate

    @property
    def frequencies(self):
        """The frequencies this transform analyzes, in Hz"""

        return self._frequencies

    @frequencies.setter
    def frequencies(self, val):

        raise ValueError("Setting frequencies outside of cwt()"
                         " is disallowed. Please use the cwt()"
                         " interface if you want to use a"
                         " different set of frequencies for"
                         " the cwt")

    @property
    def wavelet(self):
        """Returns wavelet associated with this transform object"""

        return self._wavelet

    @wavelet.setter
    def wavelet(self, wav):
        
        if wav.fs != self._fs:
            raise ValueError("Wavelet must have same sampling rate as"
                             " input data")
        if not isinstance(wav, wavedef.Wavelet):
            raise TypeError("The wavelet must be of type ghost.Wavelet")
        if wav.get_wavelet().ndim != 1:
            raise ValueError("Wavelet must be 1D")

        self._wavelet = wav

    @property
    def amplitude(self):

        return self._amplitude

    @amplitude.setter
    def amplitude(self):

        raise ValueError("Overriding the amplitude attribute"
                         " is not allowed")

    @property
    def power(self):

        return np.square(self._amplitude)

    @power.setter
    def power(self, val):

        raise ValueError("Overriding the power attribute is not"
                         " allowed")

    @property
    def time(self, val):

        return self._time

    @time.setter
    def time(self, val):

        raise ValueError("Overriding the time attribute is not"
                         " allowed")