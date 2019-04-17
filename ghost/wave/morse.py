"""This file contains an implementation of the Morse wavelet"""

import copy
import numpy as np
from scipy.special import gamma, gammaln, comb

from .wavelet import Wavelet
from . import morseutils

__all__ = ['Morse']

class Morse(Wavelet):

    def __init__(self, *, fs=None, freq=None, gamma=None, beta=None):
        """Initializes a Morse wavelet class

        Parameters
        ----------
        freq : float
            Frequency at which the wavelet reaches its peak value
            in the frequency domain, in units of Hz
        gamma : float, optional
            Gamma parameter of Morse wavelet
            Default is 3
        beta : float, optional
            Beta parameter of Morse wavelet
            Default is 20
        """

        super().__init__()

        if fs is None:
            fs = 1
        self.fs = fs

        if freq is None:
            freq = 0.25 * self.fs
        self.freq = freq  # will also set radian_freq

        # MATLAB has default time bandwidth of 60 so we have similar
        # defaults
        if gamma is None:
            gamma = 3
        if beta is None:
            beta = 20

        self.gamma = gamma
        self.beta = beta

    def get_wavelet(self, length, *, normalization=None):
        """Gets wavelet representation

        Parameters
        ----------
        length : int
            Length of wavelet
        normalization : string, optional
            The type of normalization to use, 'bandpass' or 'energy'
            If 'bandpass', the DFT of the wavelet has a peak value of 2
            If 'energy', the time-domain wavelet energy sum(abs(psi)**2)
            is 1.

        Returns
        -------
        A tuple of (psi, psif), where psi is the time-domain representation
        of the wavelet and psif is frequency-domain
        """

        if length is None:
            # Default length used for convolution, see transform.py
            length = 16384
        if length < 1:
            raise ValueError("length must at least 1 but got {}".
                             format(length))

        if normalization is None:
            normalization = 'bandpass'
        if normalization not in ('bandpass', 'energy'):
            raise ValueError("normalization must be 'bandpass' or"
                             " 'energy' but got {}".format(normalization))

        psi, psif = morseutils.morsewave(length,
                                         self._gamma,
                                         self._beta, 
                                         self._radian_freq,
                                         n_wavelets=1,
                                         normalization=normalization)

        return psi.squeeze(), psif.squeeze()

    def generate_freqs(self, N, **kwargs):

        return morseutils.morsespace(self._gamma, self._beta, N, **kwargs)

    def copy(self):

        return copy.deepcopy(self)

    def _norm_radians_to_hz(self, val):

        return val / np.pi * self._fs / 2

    def _hz_to_norm_radians(self, val):

        return val / (self._fs / 2) * np.pi

    @property
    def fs(self):

        return self._fs

    @fs.setter
    def fs(self, val):

        if not val > 0:
            raise ValueError("fs must be positive but got {}"
                             .format(val))

        self._fs = val

    @property
    def freq(self):

        return self._freq

    @freq.setter
    def freq(self, val):

        if not val > 0:
            raise ValueError("freq must be positive but got {}"
                             .format(val))

        self._freq = val
        self._radian_freq = self._hz_to_norm_radians(val)

    @property
    def radian_freq(self):

        return self._radian_freq

    @radian_freq.setter
    def radian_freq(self, val):

        if not val > 0:
            raise ValueError("radian freq must be positive but got {}"
                             .format(val))

        self._radian_freq = val
        self._freq = self._norm_radians_to_hz(val)

    @property
    def gamma(self):

        return self._gamma

    @gamma.setter
    def gamma(self, val):

        if not val > 0:
            raise ValueError("gamma must be positive")

        self._gamma = val

    @property
    def beta(self):

        return self._beta

    @beta.setter
    def beta(self, val):

        if not val > 0:
            raise ValueError("beta must be positive")

        self._beta = val

    @property
    def norm_freq(self):
        """Normalized frequency in radians,
        where Nyquist is pi"""

        return self._freq / (self._fs / 2) * np.pi

    @property
    def time_bandwidth(self):

        return self._gamma * self._beta
