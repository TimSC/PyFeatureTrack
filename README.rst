.. -*- mode: rst -*-

PyFeatureTracker
================

A python implementation of the Kanade-Lucas-Tomasi feature tracker. Some portions are optimised for cython.

Ported from the KLT C library by Stan Birchfield and Thorsten Thormaehlen, Version 1.3.4. http://www.ces.clemson.edu/~stb/klt

The code is available under the Simplified BSD License.

Dependencies
============

Python >= 2.6, PIL, Numpy, SciPy >= 0.8, cython

Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/TimSC/PyFeatureTrack.git

or if you have write privileges::

    git clone git@github.com:TimSC/PyFeatureTrack.git

Usage
-----

    python setup.py build_ext --inplace

    python example1.py
