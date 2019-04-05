"""Support for different Fourier and inverse Fourier transforms"""

import scipy.fftpack._fftpack as sff

__all__ = ['clean_scipy_cache']

def clean_scipy_cache():
    sff.destroy_zfft_cache()
    sff.destroy_zfftnd_cache()
    sff.destroy_drfft_cache()
    sff.destroy_cfft_cache()
    sff.destroy_cfftnd_cache()
    sff.destroy_rfft_cache()
    sff.destroy_ddct2_cache()
    sff.destroy_ddct1_cache()
    sff.destroy_dct2_cache()
    sff.destroy_dct1_cache()
    sff.destroy_ddst2_cache()
    sff.destroy_ddst1_cache()
    sff.destroy_dst2_cache()
    sff.destroy_dst1_cache()


