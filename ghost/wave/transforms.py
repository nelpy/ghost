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
