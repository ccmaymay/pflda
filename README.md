LittleOwl
=========

LittleOwl is a collection of software tools for fast randomized algorithms
and applications thereof.  Hoot.

LittleOwl is separated into two sub-projects, `core` and `apps`.  `core` contains a lightweight C library for randomized algorithms, called `liblowl`.  It also contains a Python interface, `pylowl`, via Cython.  `apps` contains several Python applications that use `pylowl` to do nifty things.

System Requirements
-------------------

The LittleOwl core library `liblowl` has minimal dependencies.  It is known to build on modern Linux and OS X with several versions of GCC 4.x (e.g., 4.4, 4.8).

The Python bridge `pylowl` requires modern Python 2.x (e.g., 2.6, 2.7) and Cython (e.g., 0.19).  The build is known to fail on Cython 0.11.

The `apps` library requires modern Python 2.x (e.g., 2.6, 2.7).
