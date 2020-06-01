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

    @pre.standardize_asa(x='data', fs='fs', n_signals=1, 
                         class_method=True, abscissa_vals='timestamps')
    def transform(self, data, *, timestamps=None, fs=None, freq_limits=None,
                  freqs=None, voices_per_octave=None, parallel=None,
                  verbose=None, method='ola', **kwargs):
        """Does a continuous wavelet transform.

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
        voices_per_octave : int, optional
            Number of wavelet frequencies per octave. Must be an even
            integer between 4 and 48, inclusive. Note that this parameter
            is not used if frequencies were already specified by the
            'freqs' option.
            Default is 10.
        parallel : boolean, optional
            Whether to run this function in parallel or not, using a
            single process, multithreaded model.
            Default is False.
        verbose : boolean, optional
            Whether to print messages displaying this function's progress.
            Default is False.

        Returns
        -------
        None
        """

        self.fs = fs  # does input validation, nice!
        self._time = timestamps # already sanitized by decorator

        if freqs is not None and freq_limits is not None:
            raise ValueError("freq_limits and freqs cannot both be used at the"
                             " same time. Either specify one or the either, or"
                             " leave both as unspecified")

        if voices_per_octave is None:
            voices_per_octave = 10
        if voices_per_octave not in np.arange(4, 50, step=2):
            raise ValueError("'voices_per_octave' must be an even number"
                             " between 4 and 48, inclusive")

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
            logging.warn("Module 'pyfftw' not found, using scipy backend")
            convfun = sigtools.fastconv_scipy

        # This will always return a copy, which is good because
        # we mutate the data
        input_asarray = data.squeeze().astype(np.float64)
        input_asarray -= np.mean(input_asarray)
        epoch_bounds = kwargs.pop('epoch_bounds', None)
        lengths = (np.diff(epoch_bounds, axis=1)).astype(int)

        freq_bounds_ref = self._norm_radians_to_hz(
                             self.wavelet.compute_freq_bounds(
                                 np.min(lengths)))

        if freqs is not None:
            # just in case user didn't pass in sorted
            # frequencies after all
            freqs = np.sort(freqs)
            freq_bounds = [freqs[0], freqs[1]]
            lb, ub = self._check_freq_bounds(freq_bounds, freq_bounds_ref)
            mask = np.logical_and(freqs >= lb, freqs <= ub)
            f = freqs[mask]
        elif freq_limits is not None:
            # just in case user didn't pass in limits as
            # [lower_bound, upper_bound]
            freq_limits = np.sort(freq_limits)
            freq_bounds = [freq_limits[0], freq_limits[1]]
            f_low, f_high = self._check_freq_bounds(freq_bounds, freq_bounds_ref)
        else:
            f_low = freq_bounds_ref[0]
            f_high = freq_bounds_ref[1]

        if freqs is None:
            n_octaves = np.log2(f_high / f_low)
            J = np.floor(n_octaves * voices_per_octave)
            j = np.arange(J+1)
            f = f_high / 2**(j/voices_per_octave)
        frequencies = f
        self._frequencies = np.flip(f) # Store as ascending order

        # make sure wavelet's sampling rate matches the one
        # this object uses
        self._wavelet.fs = self._fs

        wavelet_lengths = self.wavelet.compute_lengths(
                              self._hz_to_norm_radians(frequencies)).tolist()

        # Set up array as C contiguous since we will be iterating row-by-row
        out_array = np.zeros((len(frequencies), input_asarray.shape[-1]))

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

        if method == 'ola':
            if parallel:
                logging.warn(("You may not get as large a speedup as you hoped for by"
                            " running this function in parallel (single process,"
                            " multithreaded). Experimentation recommended."))
                pool = ThreadPool(cpu_count())
                # Order the frequencies such that the transform matrix starts
                # from the high frequencies at the origin
                it = [(idx, freq, length) for idx, (freq, length) in 
                        enumerate(zip(frequencies, wavelet_lengths))]
                start_time = time.time()
                pool.map(wavelet_conv, it, chunksize=1)
                pool.close()
                pool.join()
            else:
                start_time = time.time()
                # Order the frequencies such that the transform matrix starts
                # from the high frequencies at the origin
                for ii, (freq, length) in enumerate(zip(frequencies, wavelet_lengths)):
                    wavelet_conv((ii, freq, length))
        elif method == 'full':
            cwt_full(self._wavelet, self._hz_to_norm_radians(frequencies), input_asarray, out_array)
        else:
            raise ValueError(f"Invalid method type {method}")

        if verbose:
            print("Elapsed time (only wavelet convolution): {} seconds"
                  " to analyze {} frequencies".
                  format(time.time() - start_time, self._frequencies.size))

        self._amplitude = out_array

    def plot(self, *, kind=None, timescale=None, logscale=None, 
             standardize=None, relative_time=None, center_time=None,
             time_limits=None, freq_limits=None,
             ax=None, **kwargs):
        """Plots the CWT spectrogram.

        Parameters
        ----------
        kind : string, optional
            Type of spectrogram plot. Can be 'amplitude' or 'power'
            Default is 'amplitude'
        timescale : string, optional
            The time scale to use on the plot's x-axis. Can be
            'milliseconds', seconds', 'minutes', or 'hours'.
            Default is 'seconds'
        logscale : boolean, optional
            Whether to plot the frequencies axis with a log scale.
            Default is True
        standardize : boolean, optional
            Whether to plot the standardized spectrogram i.e.
            zero mean and unit standard deviation along the
            time axis
            Default is False
        relative_time : boolean, optional
            Whether the time axis shown on the plot will be relative
            Default is False
        center_time : boolean, optional
            Whether the time axis is centered around 0. This option
            is only available if 'relative_time' is se to True
            Default is False
        time_limits : nelpy.EpochArray, np.ndarray, or list, optional
            nelpy.EpochArray containing the epoch of interest, or a
            list or ndarray consisting of [lower_bound, upper_bound]
            for the times to show on the plot. Units are seconds.
            Default is show all time
        freq_limits : np.ndarray or list, optional
            ndarray or list specifying [lower_bound, upper_bound]
            for the frequencies to display on the plot. Units
            are in Hz.    
            Default is to show all frequencies
        ax : matplotlib axes object, optional
            The axes on which the plot will be drawn
        kwargs : optional
            Other keyword arguments are passed to matplotlib's
            'contourf' command.
            Example: Draw 200 contours by passing in 'levels=200'

        Returns
        -------
        ax : matplotlib axes object
            The axes on which this plot was drawn
        """

        if kind is None:
            kind = 'amplitude'
        if kind not in ('amplitude, power'):
            raise ValueError("'kind' must be 'amplitude' or 'power', but"
                             " got {}".format(kind))

        if timescale is None:
            timescale = 'seconds'
        if timescale not in ('milliseconds', 'seconds', 'minutes', 'hours'):
            raise ValueError("timescale must be 'milliseconds', seconds',"
                             " 'minutes', or 'hours' but got {}".format(timescale))

        if logscale is None:
            logscale = True
        if logscale not in (True, False):
            raise ValueError("'logscale' must be True or False but got {}".
                             format(logscale))

        if standardize is None:
            standardize = False
        if standardize not in (True, False):
            raise ValueError("'standardize' must be True or False but got {}".
                             format(standardize))

        if relative_time is None:
            relative_time = False
        if relative_time not in (True, False):
            raise ValueError("'relative_time' must be True or False but got {}".
                             format(relative_time))

        if center_time is None:
            center_time = False
        if center_time not in (True, False):
            raise ValueError("'center_time' must be True or False but got {}".
                             format(center_time))
        if center_time and not relative_time:
            raise ValueError("'relative_time' must be True to use option"
                             " 'center_time'")    

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

        if standardize:
            data = ((data - data.mean(axis=1, keepdims=True)) 
                     / data.std(axis=1, keepdims=True))

        if timescale == 'milliseconds':
            timevec = timevec * 1000
            xlabel = "Time (msec)"
        elif timescale == 'seconds':
            xlabel = "Time (sec)"
        elif timescale == 'minutes':
            timevec = timevec / 60
            xlabel = "Time (min)"
        else:
            timevec = timevec / 3600
            xlabel = "Time (hr)"

        if relative_time:
            if center_time:
                if len(timevec) & 1:
                    center_val = timevec[len(timevec)//2]
                else:
                    idx = len(timevec)//2 - 1
                    center_val = (timevec[idx] + timevec[idx]) / 2
                timevec -= center_val
            else:
                timevec -= timevec[0]

        if ax is None:
            ax = plt.gca()

        time, freq = np.meshgrid(timevec, np.flip(freqvec))
        c = ax.contourf(time, freq, data, **kwargs)
        if logscale:
            ax.set_yscale('log')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency (Hz)")

        return c

    def _norm_radians_to_hz(self, val):

        return np.array(val) / np.pi * self._fs / 2.0

    def _hz_to_norm_radians(self, val):

        return np.array(val) / (self._fs / 2.0) * np.pi

    def _check_freq_bounds(self, freq_bounds, freq_bounds_ref):

        # Inputs are Hz, outputs are Hz
        lb = freq_bounds[0]
        ub = freq_bounds[1]
        lb_ref = freq_bounds_ref[0]
        ub_ref = freq_bounds_ref[1]

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

        limits = np.atleast_1d(limits.squeeze())
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


def cwt_full(wavelet, frequencies, in_array, out_array):


    try:
        import pyfftw
    except:
        raise ImportError("This method requires pyfftw")

    if not in_array.ndim == 1:
        raise ValueError(f"Input array has {in_array.ndim} dimensions. Must be 1")

    print("Computing wavelet function for all scales...")
    N = len(in_array)
    scales = wavelet.freq_to_scale(frequencies)
    psifs = wavelet.freq_domain(N, scales)

    X = pyfftw.zeros_aligned(N, dtype='complex128')
    fft_sig = pyfftw.FFTW( X, X,
                           axes=(-1,),
                           direction='FFTW_FORWARD',
                           flags=['FFTW_ESTIMATE'],
                           threads=cpu_count())
    X[:] = in_array
    fft_sig(normalise_idft=True)


    print("Computing wavelet transform...")
    cwt_coefs = pyfftw.zeros_aligned(out_array.shape, dtype='complex128')
    fft_res = pyfftw.FFTW( cwt_coefs, cwt_coefs,
                           axes=(-1,),
                           direction='FFTW_BACKWARD',
                           flags=['FFTW_ESTIMATE'],
                           threads=cpu_count())

    cwt_coefs[:] = psifs
    cwt_coefs *= X

    fft_res(normalise_idft=True)

    out_array[:] = np.abs(cwt_coefs)





    