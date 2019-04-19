"""This module handles different input data formats"""

import logging
import numpy as np

from functools import wraps
from ..utils import get_contiguous_segments
try:
    import nelpy as nel
except:
    pass

__all__ = ['standard_format_1d_asa', 'standardize_asa']

def standardize_asa(func=None, *, x, abscissa_vals=None, fs=None,
                    n_signals=None, rowsig=None, class_method=None):
    
    logger = logging.getLogger('ghost')
    logger.setLevel(logging.INFO)

    if not isinstance(x, str):
        raise TypeError("'x' decorator argument must be a string")

    if n_signals is not None:
        try:
            if not float(n_signals).is_integer():
                raise ValueError("'n_signals' must be a positive integer")
            n_signals = int(n_signals)
        except ValueError:
            raise ValueError("'n_signals' must be a positive integer")

        if not n_signals > 0:
            raise ValueError("'n_signals' must be a positive integer")

    if abscissa_vals is not None:
        if not isinstance(abscissa_vals, str):
            raise TypeError("'abscissa_vals' decorator argument must be a string")
    if fs is not None:
        if not isinstance(fs, str):
            raise TypeError("'fs' decorator argument must be a string")

    if rowsig is None:
        rowsig = False
    if rowsig not in (True, False):
        raise ValueError("'rowsig' decorator argument must be True or False")

    if class_method is None:
        class_method = False
    if class_method not in (True, False):
        raise ValueError("'class_method' decorator argument must be True or False")

    def decorate(function):
        
        @wraps(function)
        def wrap_function(*args, **kwargs):
            
            # Intercept arguments and transform them appropriately

            kw = True
            x_ = kwargs.pop(x, None)
            fs_ = kwargs.pop(fs, None)
            abscissa_vals_ = kwargs.pop(abscissa_vals, None)

            if x_ is None:
                if class_method:
                    idx = 1
                else:
                    idx = 0
                try:
                    x_ = args[idx] # maybe x is positional argument 
                    kw = False
                except IndexError:
                    raise TypeError("{}() missing 1 required positional argument:"
                                    " '{}'".format(function.__name__, x))

            try:
                # standardize the asa
                if isinstance(x_, nel.RegularlySampledAnalogSignalArray):
                    if n_signals is not None:
                        if not x_.n_signals == n_signals:
                            raise ValueError("Input object '{}'.n_signals=={}, but expected {}".
                                              format(x, x_.n_signals, n_signals))
                            
                    if fs is not None:
                        if fs_ is not None:
                            logger.warning("'{}' was passed in, but will be overwritten" \
                                           " by '{}'s 'fs' attribute".format(fs, x_))
                        kwargs[fs] = x_.fs
                    
                    if abscissa_vals is not None:
                        if abscissa_vals_ is not None:
                            logger.warning("'{}' was passed in, but will be overwritten" \
                                           " by '{}'s 'abscissa_vals' attribute".
                                           format(abscissa_vals, x_))
                        kwargs[abscissa_vals] = x_.abscissa_vals

                    if rowsig:
                        data_ = x_._data_rowsig
                    else:
                        data_ = x_._data_colsig
                        
                    ep = np.insert(np.sum(x_.lengths), 0, 0)
                    epoch_bounds = np.hstack((ep[:-1][:, None], ep[1:][:, None]))
                    kwargs['epoch_bounds'] = epoch_bounds

                    if kw:
                        kwargs[x] = data_
                    else:
                        args = tuple([arg if ii > 0 else data_ for (ii, arg) in enumerate(args)])

                    return function(*args, **kwargs)
            
            except NameError: # nelpy not installed
                pass

            # if we're at this point, we know the input is not a nelpy object
            if not isinstance(x_, np.ndarray):
                raise TypeError("Input was not a nelpy.RegularlySampledAnalogSignalArray"
                                " so expected a numpy ndarray but got {}".format(type(x_)))
            data_ = np.atleast_1d(x_.squeeze())
            if data_.ndim == 1:
                data_ = data_.reshape((-1, 1))

            if n_signals is not None:
                if data_.shape[-1] != n_signals:
                    raise ValueError("Expected {} number of signals but got {}".
                                     format(n_signals, data_.shape[0]))

            if fs is not None:
                if fs_ is None:
                    raise TypeError("{}() missing 1 required keyword argument: '{}'".
                                    format(function.__name__, fs))
                kwargs[fs] = fs_

            if abscissa_vals is not None:
                if abscissa_vals_ is None:
                    logger.info("'{}' not passed in; generating from data".
                                   format(abscissa_vals))
                    if fs_ is None:
                        # assume default of 1
                        fs_default = 1
                        abscissa_vals_ = np.arange(data_.shape[0], dtype=np.float)/fs_default
                    else:
                        abscissa_vals_ = np.arange(data_.shape[0], dtype=np.float)/fs_
                else:
                    if not isinstance(abscissa_vals_, np.ndarray):
                        raise TypeError("Expected '{}' to be a numpy.ndarray but got {}".
                                       format(abscissa_vals, type(abscissa_vals_)))
                    if not abscissa_vals_.ndim != 1:
                        raise ValueError("'{}' should have at most one non-singleton"
                                        " dimension".format(abscissa_vals))
                    if abscissa_vals_.shape[0] != data_.shape[0]:
                        raise ValueError("The argument '{}' has {} sample points, but"
                                         " the data '{}' has {}".
                                        format(abscissa_vals, abscissa_vals_.shape[0],
                                               x, data_.shape[0]))
                    
                if fs_ is None:
                    logging.warning("'{}' not passed in; assuming default of 1 Hz".
                                    format(fs))
                    epoch_bounds = get_contiguous_segments(abscissa_vals_,
                                                            step=1/fs_default,
                                                            assume_sorted=True,
                                                            index=True,
                                                            inclusive=False)
                else:
                    epoch_bounds = get_contiguous_segments(abscissa_vals_,
                                                            step=1/fs_,
                                                            assume_sorted=False,
                                                            index=True,
                                                            inclusive=False)
                
                kwargs[abscissa_vals] = abscissa_vals_
                kwargs['epoch_bounds'] = epoch_bounds

            if kw:
                kwargs[x] = data_
            else:
                if class_method:
                    args = tuple([arg if ii != 1 else data_ for (ii, arg) in enumerate(args)])
                else:
                    args = tuple([arg if ii > 0 else data_ for (ii, arg) in enumerate(args)])

            return function(*args, **kwargs)

        return wrap_function

    # decorator called without arguments
    if func:
        return decorate(func)

    return decorate

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
                logging.warning("'Lengths' was passed in, but will be overwritten"
                                " by the nelpy object's 'lengths' attribute")
            
            return obj.data.squeeze().copy(), obj.lengths
    except NameError:
        # User doesn't have nelpy installed, continue on
        pass

    if not isinstance(obj, np.ndarray):
        raise TypeError("Input was not a nelpy.RegularlySampledAnalogSignalArray"
                        " so expected a numpy ndarray but got {}".format(type(obj)))

    # If we made it this far, we know the obj is a valid numpy array
    if obj.size == 0:
        logging.warning("Data is empty")
    if len(obj.squeeze().shape) > 1:
        raise ValueError("Numpy array must have only one dimension after being squeezed")
    if lengths is None:
        lengths = np.array([obj.size])
    if np.sum(lengths) != obj.size:
        raise ValueError("Lengths must sum to {} but got {}".format(obj.size, np.sum(lengths)))

    return obj.squeeze().copy(), lengths