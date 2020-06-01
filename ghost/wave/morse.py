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
        fs : float, optional
            The sampling rate in Hz.
            Default is 1
        freq : float, optional
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

    def __call__(self, length, *, normalization=None):
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

    def freq_domain(self, N, scales, *, derivative=False):
        
        gamma = self.gamma
        beta = self.beta
        scales = np.atleast_2d(scales).reshape((-1, 1))

        omega = np.fft.fftfreq(N) * 2 * np.pi
        x = scales * omega
        log_a = np.log(2) + (beta/gamma) * (1+np.log(gamma) - np.log(beta))
        
        H = np.zeros_like(omega)
        H[omega > 0] = 1
        
        with np.errstate(invalid='ignore', divide='ignore'):
            log_psif = log_a + beta * np.log(np.abs(x)) - np.abs(x)**gamma
            psif = np.exp(log_psif) * H

        assert np.all(np.isfinite(psif))
        
        if derivative:
            return 1j * omega * psif
        
        return psif

    def freq_to_scale(self, freqs):

        fm = np.exp( (np.log(self._beta) - np.log(self._gamma)) / self._gamma )

        # input should be in normalized radian frequencies
        freqs = np.atleast_1d(freqs)
        
        return fm / freqs

    def compute_freq_bounds(self, N, *, p=None, **kwargs):

        if p is None:
            p = 5

        wh = morseutils.morsehigh(self._gamma, self._beta, **kwargs)

        w0 = morseutils.morsefreq(self._gamma, self._beta)
        base_length = (2 * np.sqrt(2) * np.sqrt(self._gamma * self._beta)) / w0 * 4
        max_length = int(np.floor(N / p))
        max_scale = max_length / base_length
        wl = w0 / max_scale

        return [wl, wh]

    def compute_lengths(self, norm_radian_freqs):

        # Peak frequency of mother wavelet
        w0 = morseutils.morsefreq(self._gamma, self._beta)

        # 4 times the mother wavelet footprint to be safe
        # see Lilly 2017 for reference
        base_length = (2 * np.sqrt(2) * np.sqrt(self._gamma * self._beta)
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
