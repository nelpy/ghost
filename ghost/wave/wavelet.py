"""This file contains the base class definition for a wavelet"""

from abc import ABC, abstractmethod

__all__  = ['Wavelet']

class Wavelet(ABC):

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def get_wavelet(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @property
    def fc(self):
        """The center frequency is the frequency at which the
        wavelet's DFT has the highest magnitude"""
        pass