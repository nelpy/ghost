.. class:: no-web

    .. image:: https://raw.githubusercontent.com/nelpy/ghost/master/ghost-title.png
        :target: https://github.com/nelpy/ghost
        :alt: ghost-logo
        :width: 10%
        :align: right
        
| 

ghost (**G**\ rand **H**\ ub **O**\ f **S**\ pectral **T**\ ools) is the friendly phantom that helps you do signal processing on neuroscience data, especially spectral analysis ;)

It can be used as a standalone package for numpy arrays, or as a companion plugin module to nelpy. Currently it supports rudimentary wavelet analysis. Planned features include multitaper Fourier methods, phase-amplitude & phase-phase coupling, current source density analysis, and more.

ghost tries to be as lightweight yet fast as possible. Suggestions for increasing efficiency and performance are always welcome!

Installation
============

To install this package, please clone this repo and run

.. code-block:: bash

    $ python setup.py install

If you are a developer, run

.. code-block:: bash

    $ python setup.py develop

Example
=======

Suppose you have a numpy array named ``X`` which was sampled at 1 kHz, on which you want to run a continuous wavelet transform:

.. code-block:: python

    from ghost.wave import ContinuousWaveletTransform
    
    cwt = ContinuousWaveletTransform()
    cwt.transform(X, fs=1000)
    
If you have a nelpy ``AnalogSignalArray`` named ``asa``, you can simply do:

.. code-block:: python

    from ghost.wave import ContinuousWaveletTransform
    
    cwt = ContinuousWaveletTransform()
    cwt.transform(asa)
    
In either case, you can obtain the spectrogram by calling ``plot``:

.. code-block:: python

    cwt.plot(logscale=False, 
             standardize=True, 
             cmap=plt.cm.Spectral_r,
             levels=300, 
             vmin=0, 
             vmax=10)

And that's it! Short and simple.
        
