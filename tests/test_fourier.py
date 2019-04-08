import ghost.sigtools
import numpy as np

def test_chirpz_dft():

    # Test even and odd (prime!) lengths
    x = np.random.rand(1009)
    x_np = np.fft.fft(x)
    x_chirpz = ghost.sigtools.chirpz_dft(x)

    y = np.random.rand(1010)
    y_np = np.fft.fft(y)
    y_chirpz = ghost.sigtools.chirpz_dft(y)

    assert np.allclose(x_np, x_chirpz)
    assert np.allclose(y_np, y_chirpz)