"""
Helper functions to generate Morse wavelets
"""

import numpy as np
from scipy.special import gammaln, comb
from scipy.special import gamma as gammafunc

__all__ = ['morsewave', 'morseafunc', 'morsefreq', 'morsemom',
           'morsef', 'morsespace', 'morsehigh', 'morselow',
           'morseprops']

# Note: The functions here are arranged in terms of locality
# so that 'bottom-level' methods are close to the 'top-level'
# methods. An alternative organization to group all private
# methods (starting with underscore) together but I think
# it makes more sense this way.
# These functions are preferably accessed through a Morse
# class instance but can be used directly from here if
# desired

def morsewave(N, gamma, beta, freqs, *, n_wavelets=None,
              normalization=None):
    """
    Parameters
    ----------
    N : integer
        Length of each wavelet
    gamma : float
        Gamma parameter for the Morse wavelet
    beta : float
        Beta parameter for the Morse wavelet
    freqs : a scalar or a 1-D array
        The radian frequencies at which the Fourier transform of
        the wavelets reach their maximum amplitudes
    n_wavelets : int
        Number of different of orthogonal wavelets
        Default is 1.
    normalization : string, optional
        Normalization type.
        If 'bandpass', the DFT of the wavelet has a peak value of 2
        If 'energy', the time-domain wavelet energy sum(abs(psi)**2)
        is 1 for each frequency requested
        Default is 'bandpass'

    Returns
    -------
    psi: np.ndarray, with shape (N, len(center_freqs), n_wavelets)
        Morse wavelet in time domain, specified by beta and gamma 
    psif: np.ndarray, with shape (N, len(center_freqs), n_wavelets)
        Morse wavelet in frequency domain

    References
    ----------
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsewave.m
    """

    _check_gamma_beta(gamma, beta)

    freqs = np.atleast_1d(np.array(freqs).squeeze())
    if freqs.ndim != 1:
        raise ValueError("Freqs must be either a scalar or an array"
                         " with one non-singleton dimension")

    if n_wavelets is None:
        n_wavelets = 1
    if not n_wavelets > 0:
        raise ValueError("n_wavelets must be positive")
    
    if normalization is None:
        normalization = 'bandpass'
    if normalization not in ('bandpass', 'energy'):
        raise ValueError("Normalization must be 'energy' or 'bandpass'")

    #TODO: Implement edge family when no longer experimental
    
    psi  = np.zeros((N, freqs.size, n_wavelets), dtype=complex)
    psif = np.zeros((N, freqs.size, n_wavelets))

    for ii, freq in enumerate(freqs):
        psi[:, ii:ii+1, :], psif[:, ii:ii+1, :] = _morsewave(N, n_wavelets, 
                                                      gamma, beta,
                                                      abs(freq),
                                                      normalization=normalization)
        if freq < 0:
            if psi.size > 0:
                psi[:, ii, :] = psi[:, ii, :].conj()
            psif[1:, ii, :] = np.flip(psif[1:, ii, :], axis=0)
    
    return psi, psif

def _morsewave(N, n_wavelets, gamma, beta, freqs, *, normalization=None):

    # This has identical arguments as morsewave. It was probably the
    # first version of the function.

    _check_gamma_beta(gamma, beta)

    freqs = np.atleast_1d(np.array(freqs).squeeze())
    if freqs.ndim != 1:
        raise ValueError("Freqs must be either a scalar or an array"
                         " with one non-singleton dimension")

    if n_wavelets is None:
        n_wavelets = 1
    if not n_wavelets > 0:
        raise ValueError("n_wavelets must be positive")

    if normalization is None:
        normalization = 'bandpass'
    if normalization not in ('bandpass', 'energy'):
        raise ValueError("Normalization must be 'bandpass', or 'energy'")

    f0 = morsefreq(gamma, beta, nout=1)
    fact = freqs / f0
    w = 2 * np.pi * np.linspace(0, 1-1/N, N) / fact

    if normalization == 'energy':
        if beta == 0:
            psizero = np.exp(-w ** gamma)
        else:
            psizero = np.exp(beta ** np.log(w) - w ** gamma)
    else:
        if beta == 0:
            psizero = 2 * np.exp(-w ** gamma)
        else:
            # Alternate calculation to cancel things that blow up
            with np.errstate(divide='ignore', invalid='ignore'):
                psizero = 2 * np.exp(- beta * np.log(f0) + f0 ** gamma 
                                    + beta * np.log(w) - w ** gamma)

    psizero[0] /= 2 # due to unit step function

    psizero[psizero == np.nan] = 0

    X = _morsewave_first_family(fact, N, n_wavelets, gamma, beta, 
                                w, psizero, normalization=normalization)

    # TODO: Implement edge wavelet when no longer experimental

    X[X == np.inf] = 0
    X[X == np.nan] = 0

    wmat = np.broadcast_to(w[:, None, None], (len(w), 1, n_wavelets))

    Xr = X * np.exp(1j * wmat * (N + 1) / 2 * fact) # ensures wavelets are centered
    
    x = np.fft.ifft(Xr, axis=0)

    return x, X

def _morsewave_first_family(fact, N, n_wavelets, gamma, beta,
                            w, psizero, *, normalization=None):

    _check_gamma_beta(gamma, beta)

    w = np.atleast_1d(np.array(w).squeeze())
    psizero = np.atleast_1d(np.array(psizero).squeeze())

    if w.shape != psizero.shape:
        raise ValueError("w must have the same shape as psizero")

    if w.ndim != 1:
        raise ValueError("w must have only one non-singleton dimension")
    
    if psizero.ndim != 1:
        raise ValueError("Psizero must have only one non-singleton dimension")

    if normalization is None:
        normalization = 'bandpass'
    if normalization not in ('bandpass', 'energy'):
        raise ValueError("Normalization must be 'bandpass', or 'energy'")

    r = (2 * beta + 1) / gamma
    c = r - 1
    L = 0 * w
    index = np.arange(round(N / 2)).astype(int)
    psif = np.zeros((len(psizero), 1, n_wavelets))

    for nn in range(n_wavelets):
        if normalization == 'energy':
            A = morseafunc(gamma, beta, order=nn + 1,
                           normalization=normalization)
            coeff = np.sqrt(1 / fact) * A
        else:
            if beta != 0:
                coeff = np.sqrt(np.exp(gammaln(r)
                                       + gammaln(nn + 1)
                                       - gammaln(nn + r)))
            else:
                coeff = 1

        L[index] = _laguerre(2 * w[index] ** gamma, nn, c)

        psif[:,0,nn] = coeff * psizero * L

    return psif

def morseafunc(gamma, beta, *, normalization=None, order=None):
    """Returns the generalized Morse wavelet amplitude or a-function.

    Parameters
    ----------
    gamma : float
        Gamma parameter of the Morse wavelet
    beta : float
        Beta parameter of the Morse wavelet
    normalization : string, optional
        Type of normalization to use.
        If 'bandpass', then A is chosen such that the maximum of
        the frequency-domain wavelet is equal to 2.
        If 'energy', then the sum of the wavelet's squared coefficients
        will be 1.
        Default is 'bandpass'
    order : int, optional
        The order of the wavelet. Note that this parameter is only used
        for the unit energy normalization

    References
    ----------
    - Lilly and Olhede (2009).  Higher-order properties of analytic wavelets.
    IEEE Trans. Sig. Proc., 57 (1), 146--160.
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morseafun.m
    """

    _check_gamma_beta(gamma, beta)

    if normalization is None:
        normalization = 'bandpass'
    if normalization not in ('bandpass', 'energy'):
        raise ValueError("Normalization must be 'bandpass' or 'energy'")

    if order is None:
        order = 1
    if order < 0:
        raise ValueError("Order must non-negative")

    if normalization =='bandpass':
        w_p = morsefreq(gamma, beta, nout=1)
        if beta == 0:
            a = 2
        else:
            a = 2 / np.exp(beta * np.log(w_p) - w_p ** gamma)
    elif normalization == 'test':
        r = (2 * beta + 1) / gamma
        a = np.sqrt( (2 * np.pi * gamma * 2 ** r) /  gammafunc(r) )
    elif normalization == 'energy':
        r = (2 * beta + 1) / gamma
        a = np.sqrt( 2 * np.pi * gamma * (2 ** r) 
                     * np.exp(gammaln(order) - gammaln( order + r - 1)) )
    
    return a

def _laguerre(x, k, c):
    """
    Generalized Laguerre polynomials
    """
    x = np.atleast_1d(np.array(x).squeeze())

    if x.ndim != 1:
        raise ValueError("The input x must have only one"
                         " non-singleton dimension")

    y = (0 * x).astype(np.float64)
    for m in range(k+1):
        fact = np.exp(gammaln(k + c + 1)
                      - gammaln(c + m + 1)
                      - gammaln(k - m + 1))
        y += np.power(-1, m) * fact * (x ** m) / gammafunc(m + 1)

    return y

def morsefreq(gamma, beta, *, nout=None):
    """
    Calculate important frequencies for generalized Morse wavelets
    
    Parameters
    ----------
    gamma : float
        Gamma parameter of the Morse wavelet
    beta : float
        Beta parameter of the Morse wavelet
    nout: int, optional
        Number of outputs
        Default is 1.
    
    Returns
    -------
    fm : float
        The modal or peak frequency, in radians
    fe : float
        The "energy" frequency, in radians
    fi : float
        The instantaneous frequency at the wavelet center, in radians
    cf : float
        Curvature of fi, in radians

    References
    ----------
    - Lilly and Olhede (2009).  Higher-order properties of analytic wavelets.  
    IEEE Trans. Sig. Proc., 57 (1), 146--160.
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsefreq.m
    """

    _check_gamma_beta(gamma, beta)

    if nout is None:
        nout = 1
    if not nout in (1, 2, 3, 4):
        raise ValueError("nout must be 1, 2, 3, or 4")

    fm = np.exp( (np.log(beta) - np.log(gamma)) / gamma )
    
    if nout > 1:
        fe = (1 / (2 ** (1 / gamma))
              * gammafunc((2 * beta + 2) / gamma)
              / gammafunc((2 * beta + 1) / gamma))

    if nout > 2:
        fi = gammafunc((beta + 2) / gamma)  / gammafunc((beta + 1) / gamma)

    if nout > 3:
        m2, n2, k2 = morsemom(2, gamma, beta, nout=3)
        m3, n3, k3 = morsemom(3, gamma, beta, nout=3)
        cf = - k3 / np.sqrt(k2 ** 3)

    if nout == 1:
        return fm
    elif nout == 2:
        return fm, fe
    elif nout == 3:
        return fm, fe, fi
    else:
        return fm, fe, fi, cf

def morsemom(p, gamma, beta, *, nout=None):
    """Frequency-domain moments of generalized Morse wavelets.

    Note that all outputs are evaluated using the bandpass
    normalization, which has max(abs(psi(omega)))=2.

    Parameters
    ----------
    p : int
        The order of the frequency-domain moment to compute
    gamma : float
        The gamma parameter of the Morse wavelet
    beta : float
        The beta parameter of the Morse wavelet
    nout : int, optional
        Number of outputs to return

    Returns
    -------
    m : float
        The pth moment defined by
        mp = 1/(2pi) int omega^p psi(omega) d omega ,
        where omega is in radians
    n : float
        The pth energy moment defined by
        np = 1/(2pi) int omega^p |psi(omega)|.^2 d omega ,
        where omega is in radians
    k : float
        The pth order cumulant
    l : float
        The pth order energy cumulant

    References
    ----------
    - Lilly and Olhede (2009).  Higher-order properties of analytic wavelets.  
    IEEE Trans. Sig. Proc., 57 (1), 146--160.
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsemom.m
    """

    _check_gamma_beta(gamma, beta)

    if p < 0:
        raise ValueError("p must be non-negative")

    if nout is None:
        nout = 1
    if nout not in (1, 2, 3, 4):
        raise ValueError("nout must be 1, 2, 3, or 4")

    m = _morsemom1(p, gamma, beta)

    if nout > 1:
        n = 2 / ( 2**((1 + p)/gamma) ) * _morsemom1(p, gamma, 2 * beta)

    if nout > 2:

        moments = _morsemom1(np.arange(p + 1), gamma, beta)
        cumulants = _moments_to_cumulants(moments)
        k = cumulants[p]

    if nout > 3:
        moments = (2 / ( 2 ** ( (1 + np.arange(p + 1)) / gamma ))
                   * _morsemom1(np.arange(p+1), gamma, 2 * beta))
        cumulants = _moments_to_cumulants(moments)
        l = cumulants[p]

    if nout == 1:
        return m
    elif nout == 2:
        return m, n
    elif nout == 3:
        return m, n, k
    else:
        return m, n, k, l

def _morsemom1(p, gamma, beta):

    return morseafunc(gamma, beta) * morsef(gamma, beta + p)

def _moments_to_cumulants(moments):
    """Converts the first N moments contained into the first N cumulants

    Parameters
    ----------
    moments : float or np.ndarray

    Returns
    -------
    Array containing the cumulants
    """

    moments = np.atleast_1d(np.array(moments).squeeze())

    if moments.ndim != 1:
        raise ValueError("Moments must be either a scalar or array with only one"
                         " non-singleton dimension")

    cumulants = np.zeros(moments.size)
    cumulants[0] = np.log(moments[0])

    for n in range(1, len(moments)):
        coeff = 0
        for k in range(1, n):
            coeff += (comb(n-1, k-1) * cumulants[k]
                      * moments[n-k] / moments[0])
        cumulants[n] = moments[n] / moments[0] - coeff

    return cumulants

def morsef(gamma, beta):
    """Computes the generalized Morse wavelet first moment 'f'

    Parameters
    ----------
    gamma : float or np.ndarrray
        Gamma parameter of the Morse wavelet
    beta : float or np.ndarray
        Beta Parameter of the Morse wavelet
        
    Returns
    -------
    The normalized first frequency-domain moment F_{gamma, beta} of
    the lower-order generalized Morse wavelet specified by parameters
    gamma and beta

    References
    ----------
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsemom.m
    """

    return 1 / (2 * np.pi * gamma) * gammafunc((beta + 1) / gamma)

def morsespace(gamma, beta, N, *, high=None, eta=None, 
               pack_num=None, low=None, density=None):
    """Logarithmically-spaced frequencies for generalized Morse wavelets.
    
    Parameters
    ----------
    gamma : float
        Gamma parameter of the Morse wavelet
    beta : float
        Beta parameter of the Morse wavelet
    N : int
        Length of the wavelet
    eta : float, optional
        The ratio of a frequency-domain wavelet at the Nyquist frequency
        to its peak value. Must be between 0 and 1.
        Default is 0.1, which means the highest-frequency wavelet will
        decay to at least 10% of its peak value at the Nyquist frequency
    high : float, optional
        Frequency, in radians per sample point. This value is used along
        with eta. Specifically the highest frequency is the minimum
        of the specified value 'high' and the largest frequency for which
        the wavelet will satisfy the threshold level eta.
    pack_num : float, optional
        The r-level/packing number. This means the lowest frequency
        wavelet will reach some number pack_num times its central
        window width at the ends of the time series.
        A choice of pack_num=1 corresponds to 
        roughly 95% of the time-domain wavelet energy being contained
        within the time series endpoints for a wavelet at the center
        of the domain.
        Default is 5, which means at the lowest frequency, five
        wavelets will fit into the time series, with 5% energy overlap
        between them
    low : float, optional
        Frequency, in radians per sample point. This value is used along
        with pack_num. Specifically, the lowest frequency is the
        maximum of the specified value 'low' and the frequency determined
        by 'pack_num'
    density : float, optional
        Density. Higher values of mean more overlap in the frequency
        domain. density=1 means that the peak of one wavelet is located
        at the half-power point of the adjacent wavelet. density=4 means
        that four other wavelets will occur between the peak of one
        wavelet and its half-power point.
        Default is 4.

    References
    ----------
    - Lilly (2017). Element analysis: a wavelet-based method for analysing
    time-localized events in noisy time series. Proceedings of the Royal
    Society A. Volume 473: 20160776, 2017, pp. 1–28
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsespace.m
    """

    _check_gamma_beta(gamma, beta)

    if N < 2:
        raise ValueError("N must be at least 2")

    if eta is None:
        eta = 0.1
    if eta < 0 or eta > 1:
        raise ValueError("eta must be between 0 and 1")

    if high is None:
        high = np.pi
    if high < 0 or high > np.pi:
        raise ValueError("high must be between 0 and pi")

    if pack_num is None:
        pack_num = 5
    if not pack_num > 0:
        raise ValueError("pack_num must be positive")

    if low is None:
        low = 0
    if low < 0 or low > np.pi:
        raise ValueError("low must be between 0 and pi")
    
    if density is None:
        density = 2
    if not density > 0:
        raise ValueError("density must be positive")

    high_freq = min(high, morsehigh(gamma, beta, eta))
    low_freq  = max(low,  morselow(gamma, beta, pack_num, N))

    voices_per_octave = 4
    # dimensionless window width
    p, _, _ = morseprops(gamma, beta)

    r = 1 + 1/(density * p)
    print(low_freq)
    N = int(np.floor( np.log(high_freq/low_freq) / np.log(r) ))
    #N = int(np.log2(high/low) * voices_per_octave)
    freqs = high_freq * np.ones(N+1) / (r**np.arange(0, N+1))

    # so that we returning ascending order
    return np.flip(freqs)

def morsehigh(gamma, beta, eta=None):
    """High-frequency cutoff of the generalized Morse wavelets specified
    by gamma and beta, with cutoff level eta

    This function gives the highest possible radian frequency for which
    the generalized Morse wavelet will have greater than eta times its
    peak value at the Nyquist frequency. 

    If f = morsefreq(gamma, beta) is the wavelet peak frequency, then
        psi (pi * f / fhigh) <= eta * psi(f)
    is the cutoff condition

    Parameters
    ----------
    gamma : float
        The gamma parameter of the Morse wavelet
    beta : float
        The beta parameter of the Morse wavelet
    eta : float, optional
        The ratio of the frequency-domain wavelet at Nyquist
        compared to its peak value.
        Default is 0.1

    References
    ----------
    - Lilly (2017). Element analysis: a wavelet-based method for analysing
    time-localized events in noisy time series. Proceedings of the Royal
    Society A. Volume 473: 20160776, 2017, pp. 1–28
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsehigh.m
    """

    _check_gamma_beta(gamma, beta)

    if eta is None:
        eta = 0.1
    if eta < 0 or eta > 1:
        raise ValueError("eta must be between 0 and 1")

    N = 10000
    w_high = np.linspace(1e-12, np.pi, N)
    
    w = morsefreq(gamma, beta) * np.pi / w_high

    #  Use logs to avoid errors for really small gammas
    #  Note that ln(2)'s cancel
    lnpsi1 = (beta/gamma) * np.log(np.exp(1)* gamma / beta)
    lnpsi2 = beta * np.log(w) - w ** gamma
    lnpsi = lnpsi1 + lnpsi2
    idx = np.argwhere(np.log(eta) - lnpsi < 0).squeeze()[0]

    return w_high[idx]

def morselow(gamma, beta, pack_num, N):
    """Low frequency cutoff of the generalized Morse wavelet specified
    by gamma and beta, with cutoff determined by the r level and length
    N of the wavelet.

    The cutoff frequency is chosen such that the lowest-frequency wavelet
    will reach some number r times its central window width at the ends
    of the time series. 

    Parameters
    ----------
    gamma : float
        Gamma parameter of the Morse wavelet
    beta : float
        Beta parameter of the Morse wavelet
    pack_num : float
        The r-level/packing number. A choice of r=1 corresponds to 
        roughly 95% of the time-domain wavelet energy being contained
        within the time series endpoints for a wavelet at the center
        of the domain.
    N : float
        The length of the wavelet

    References
    ----------
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsespace.m
    """

    _check_gamma_beta(gamma, beta)

    if pack_num is None:
        pack_num = 5
    if not pack_num > 0:
        raise ValueError("pack_num must be positive")

    if N < 2:
        raise ValueError("N must be at least 2")

    p, _, _ = morseprops(gamma, beta)

    return 2 * np.sqrt(2) * p * pack_num / N

def morseprops(gamma, beta):
    """Computes properties of the demodulated generalized Morse wavelets

    Parameters
    ----------
    gamma : float
        Gamma parameter of the Morse wavelet
    beta : float
        Beta parameter of the Morse wavelet
    
    Returns
    -------
    p : float
        The dimensionless time-domain window width
    skew : float
        Imaginary part of the time-domain demodulate, or 'demodulate
        skewness'
    kurt : float
        Normalized fourth moment of the time-domain demodulate, or
        'demodulate kurtosis'

    References
    ----------
    Lilly & Olhede (2008b)
    - JLAB (C) 2004--2016 J. M. Lilly and F. Rekibi
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morseprops.m
    """

    _check_gamma_beta(gamma, beta)

    p = np.sqrt(gamma * beta)
    skew = (gamma - 3) / beta
    kurt = 3 - np.square(skew) - 2 / np.square(p)
    
    return p, skew, kurt

def _check_gamma_beta(gamma, beta):

    if gamma < 0:
        raise ValueError("Gamma must be positive")

    if beta < 0:
        raise ValueError("Beta must be positive")