=====
Ghost
=====

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

Suppose you have a numpy array ``X`` on which you want to run a continuous wavelet transform:

.. code-block:: python

    import ghost.wave
    
    wt = ghost.wave.ContinuousWaveletTransform()
    fs = 1000
    spec = wt.cwt(X, fs=fs)

If you have a nelpy ``AnalogSignalArray``, you can do:

.. code-block:: python

    import ghost.wave
    
    wt = ghost.wave.ContinuousWaveletTransform()
    spec = wt.cwt(X, fs=X.fs, output='asa')

And that's it! Short and simple.
