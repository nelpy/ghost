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
        self.freq = freq  # will also set norm_radian_freq

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
                                         self._norm_radian_freq,
                                         n_wavelets=1,
                                         normalization=normalization)

        return psi.squeeze(), psif.squeeze()

    def generate_omegas(self, N, **kwargs):

        return morseutils.morsespace(self._gamma, self._beta, N, **kwargs)

    def compute_lengths(self, norm_radian_freqs):

        # Peak frequency of mother wavelet
        w0 = morseutils.morsefreq(self._gamma, self._beta)

        # 4 times the mother wavelet footprint to be safe
        # see Lilly 2017 for reference
        base_length = (2 * np.sqrt(2) * self._gamma * self._beta
                        / w0 * 4)

        scale_fact = w0 / norm_radian_freqs

        # lower frequencies require more samples
        # and higher frequencies fewer
        return np.ceil(scale_fact * base_length).astype(int)

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
    def frequency(self):

        return self._freq

    @frequency.setter
    def frequency(self, val):

        if not val > 0 and val <= self._fs/2:
            raise ValueError("The frequency must be between 0"
                             " and the Nyquist frequency {} Hz"
                             " but got {}".format(self._fs / 2, val))

        self._freq = val
        self._norm_radian_freq = self._hz_to_norm_radians(val)

    @property
    def norm_radian_freq(self):

        return self._norm_radian_freq

    @norm_radian_freq.setter
    def norm_radian_freq(self, val):

        if not val > 0 and val <= np.pi:
            raise ValueError("The normalized radian frequency must"
                             " be between 0 and the Nyquist frequency"
                             " pi but got {]".format(val))

        self._norm_radian_freq = val
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
    def time_bandwidth(self):

        return self._gamma * self._beta
