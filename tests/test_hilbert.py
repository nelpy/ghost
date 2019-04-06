import ghost.sigtools
import numpy as np

def test_hilbert():

    siglen = 30000*60

    x = np.random.rand(siglen)
    analytic_scipy = ghost.sigtools.analytic_signal_scipy(x)
    analytic_fftw = ghost.sigtools.analytic_signal_fftw(x)

    assert np.allclose(analytic_scipy, analytic_fftw)