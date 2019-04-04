"""This module handles different input data formats"""

import warnings
import numpy as np
try:
    import nelpy as nel
except:
    pass

__all__ = ['standard_format_1d_asa']

def standard_format_1d_asa(obj, *, lengths=None):
    """This function takes in an object and outputs a numpy array along with
       the lengths, which specify the blocksize (in elements) of the data
       that should be treated as a contiguous segment.

    Parameters
    ----------
    obj : numpy.ndarray or nelpy.RegularlySampledAnalogSignalArray (or a
          subclass thereof)
        Input data object
    lengths : array-like
        Array-like object that specifies the blocksize (in elements) of
        the data that should be treated as a contiguous segment

    Returns
    -------
    out : numpy.ndarray
        A 1D numpy array
    lengths : array-like
        Array-like object that specifies the blocksize (in elements) of
        the data that should be treated as a contiguous segment     
    """

    try:
        if isinstance(obj, nel.RegularlySampledAnalogSignalArray):
            if obj.n_signals != 1:
                raise ValueError("Input object must have only one signal")

            if lengths is not None:
                warnings.warn("'Lengths' was passed in, but will be overwritten",
                                " by the nelpy object's 'lengths' attribute")
            
            return obj.data.squeeze().copy(), obj.lengths
    except NameError:
        # User doesn't have nelpy installed, continue on
        pass

    if not isinstance(obj, np.ndarray):
        raise TypeError("Input was not a nelpy.RegularlySampledAnalogSignalArray"
                        " so expected a numpy ndarray but got {}".format(type(obj)))

    # If we made it this far, we know the obj is a valid numpy array
    out = obj.squeeze()
    if out.ndim != 1:
        raise ValueError("Numpy array must have only one dimension after being squeezed")
    if lengths is None:
        lengths = np.array([len(out)])
    if np.sum(lengths) != out.size:
        raise ValueError("Lengths must sum to {} but got {}".format(out.size, np.sum(lengths)))

    return out.copy(), lengths