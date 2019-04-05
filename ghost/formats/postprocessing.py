"""This module handles the output types a user wants"""

import numpy as np
import logging

try:
    import nelpy as nel
except:
    pass

__all__ = ['output_numpy_or_asa']

def output_numpy_or_asa(obj, data, *, output_type=None, labels=None):
    """This function returns a numpy ndarray or nelpy.AnalogSignalArray

    Parameters
    ----------
    obj : numpy.ndarray or a nelpy object
    data : numpy.ndarray, with shape (n_samples, n_signals)
        Data is either passed through as the np.ndarray
        or used to form a nelpy object, depending on 'output_type'.
    output_type : string, optional
        Specifies the object that should be returned.
        Default is a numpy np.ndarray
    labels : np.adarray of string, optional
        Labels that will be attached to the nelpy object, if
        that is the desired output type. If the output type is
        'numpy', the labels are ignored.
    
    Returns
    -------
    Output object of the specified type. If a numpy array, it will
    have shape (n_samples, n_signals)
    """

    if data.size == 0:
        logging.warning("Output data is empty")
    
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy ndarray")
    if output_type is not None:
        if output_type != 'asa':
            raise TypeError(("Invalid output type {} specified".
                             format(output_type)))

    if output_type == 'asa':
        try:
            res = isinstance(obj, nel.RegularlySampledAnalogSignalArray)

            if res is False:
                raise TypeError("You specified output type {} but the input"
                                " object was not a nelpy object. Cannot form an"
                                " ASA around the input object".format(output_type))
            # Transpose data since ASAs have shape (n_signals, n_samples)
            out = nel.AnalogSignalArray(data.T,
                                abscissa_vals=obj.abscissa_vals,
                                fs=obj.fs,
                                support=obj.support, 
                                labels=labels)
            return out
        except NameError:
            raise ModuleNotFoundError("You must have nelpy installed for"
                                          " output type {}".format(output_type))
    
    return data
