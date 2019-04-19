import numpy as np

def get_contiguous_segments(data, *, step=None, assume_sorted=None,
                            index=False, inclusive=False):
    """Compute contiguous segments (separated by step) in a list.
    """

    if inclusive:
        assert index, "option 'inclusive' can only be used with 'index=True'"
    data = np.asarray(data)

    if not assume_sorted:
        if not is_sorted(data):
            data = np.sort(data)  # algorithm assumes sorted list

    if step is None:
        step = np.median(np.diff(data))

    # assuming that data(t1) is sampled somewhere on [t, t+1/fs) we have a 'continuous' signal as long as
    # data(t2 = t1+1/fs) is sampled somewhere on [t+1/fs, t+2/fs). In the most extreme case, it could happen
    # that t1 = t and t2 = t + 2/fs, i.e. a difference of 2 steps.

    chunk_size = 1000000
    stop = data.size
    breaks = []
    for chunk_start in range(0, stop, chunk_size):
        chunk_stop = int(min(stop, chunk_start + chunk_size + 2))
        breaks_in_chunk = chunk_start + np.argwhere(np.diff(data[chunk_start:chunk_stop])>=2*step)
        if np.any(breaks_in_chunk):
            breaks.extend(breaks_in_chunk)
    breaks = np.array(breaks)
    starts = np.insert(breaks+1, 0, 0).astype(int)
    stops = np.append(breaks, len(data)-1).astype(int)
    bdries = np.vstack((data[starts], data[stops] + step)).T 
    if index:
        if inclusive:
            indices = np.vstack((starts, stops)).T
        else:
            indices = np.vstack((starts, stops + 1)).T
        return indices
    return np.asarray(bdries)

def is_sorted(x, chunk_size=None):
    """Returns True if iterable is monotonic increasing (sorted).
    NOTE: intended for 1D array, list or tuple. Will not work on
    more than 1D
    This function works in-core with a modest memory footprint.
    chunk_size = 100000 is probably a good choice.
    """

    if not isinstance(x, (tuple, list, np.ndarray)):
        raise TypeError("Unsupported type {}".format(type(x)))

    x = np.atleast_1d(np.array(x).squeeze())
    if x.ndim > 1:
            raise ValueError("Input x must have only one non-singleton"
                             " dimension")

    if chunk_size is None:
        chunk_size = 500000
    stop = x.size
    for chunk_start in range(0, stop, chunk_size):
        chunk_stop = int(min(stop, chunk_start + chunk_size + 1))
        chunk = x[chunk_start:chunk_stop]
        if not np.all(chunk[:-1] <= chunk[1:]):
            return False
    return True