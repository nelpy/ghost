"""This file contains an implementation of the Morlet wavelet"""

import numpy as np
from .wavelet import Wavelet
from scipy.misc import logsumexp
import copy

__all__ = ['Morlet']

class Morlet(Wavelet):

    def __init__(self, *, w0=None, freq=None, fs=None):
        """Initializes a Morlet wavelet class

        Parameters
        ==========
        w0 : float, optional
            The non-dimensional frequency parameter. Should be > 5
            for wavelet admissibility. 
            Default is 6.
        freq : float, optional
            The frequency where the DFT of the wavelet reaches its
            maximum magnitude. Units in Hz. 
            Default is 1 Hz.
        fs : float, optional
            The sampling rate at which to generate the wavelet, in Hz.
            Default is 1 Hz.
        """   
        
        super().__init__()

        # TODO: Input validation
        if w0 is None:
            w0 = 6
        if freq is None:
            freq = 1
        if fs is None:
            fs = 1

        self._w0 = w0
        self._fs = fs
        self._freq = freq
        self._scale = None
        self._time_repr = None
        
        self._freq_to_scale()
        self._compute_time_repr()

    def get_wavelet(self):

        return self._time_repr

    def _freq_to_scale(self):
        period = 1/self._freq
        self._scale = period * (self._w0 + np.sqrt(2 + self._w0**2)) / (4*np.pi)

    def _compute_time_repr(self):

        self._freq_to_scale()
        
        dt = 1/self._fs
        # Capture the highest possible frequency
        # in the data
        M = self._fs
        # times to use, centered at zero
        t = np.arange(-(M+1)/2, (M+1)/2) * dt
        
        # Assigning eta and w0 is mainly to make the code
        # cleaner to read
        eta = t / self._scale 
        w0 = self._w0

        # Sample wavelet and normalize
        scaledwav = np.pi**(-0.25) * np.exp(-1/2*eta**2) * (np.exp(1j*w0*eta) - np.exp(-1/2*w0**2))
        normconst = (dt / self._scale) ** .5
        
        self._time_repr =  scaledwav * normconst

    def _recompute(self):
        self._compute_time_repr()

    def copy(self):
        return copy.deepcopy(self)

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, sample_rate):
        self._fs = sample_rate
        self._recompute()

    @property
    def w0(self):
        return self._w0

    @w0.setter
    def w0(self, norm_freq):
        self._w0 = norm_freq
        self._recompute()

    @property
    def freq(self):

        return self._freq

    @freq.setter
    def freq(self, freq):
        self._freq = freq
        self._recompute()
