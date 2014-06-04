LittleOwl
=========

LittleOwl is a collection of software tools for fast randomized algorithms and applications thereof.  Hoot.

LittleOwl consists of two main components, the `lowl` and `pylowl` libraries, and several sub-projects in the `proj` directory, including a particle filter for streaming learning in LDA.  The `lowl` library (or `liblowl`) is a lightweight C library implementing an assortment of randomized algorithms; `pylowl` is a Python extension module wrapping `liblowl` via Cython.

System Requirements
-------------------

The `lowl` library itself has minimal dependencies.  It is known to build on modern Linux and OS X with several versions of GCC 4.x (e.g., 4.4, 4.8).

The Python extension module `pylowl` requires modern Python 2.x (e.g., 2.6, 2.7) and Cython (e.g., 0.19).  The build is known to fail on Cython 0.11.

The projects under `proj` have varying dependencies; however, most of them depend at least on `pylowl`, and as such require at minimum a modern Python 2.x (e.g., 2.6, 2.7).  (GCC and Cython are not necessarily required if you can convince someone to make a binary distribution for you.)

Building and Installing
-----------------------

We use `scons` as a build system.  This requires only a modern Python distribution; `scons` itself is provided with `littleowl` for your convenience.  To build and test `lowl` and `pylowl` (from the repository root):

```
$ python scons.py lowl-tests pylowl-tests
```

To install `lowl` locally:

```
$ python scons.py --prefix=$HOME lowl-install
```

The `--user` flag is identical in this context:

```
$ python scons.py --user lowl-install
```

To install `pylowl` in the user installation scheme (location in home directory that is automatically in your Python module search path):

```
$ python scons.py --user pylowl-install
```

To build the Python extension module implementing a particle filter for LDA:

```
$ python scons.py pflda
```

The installation targets are decoupled; they do not depend on one another.  Thus, to install `lowl`, `pylowl`, and `pflda` locally, you would need to do:

```
$ python scons.py --user lowl-install pylowl-install pflda-install
```

Scons expects to be run from the root of the repository; however, you may run from a subdirectory by passing the `-u` flag to scons.  E.g., building `pflda` from the `src/proj/pflda` subdirectory:

```
$ pushd src/proj/pflda
$ python ../../../scons.py -u pflda
$ popd
```

(This makes more sense if you have [installed scons](http://scons.org/download.php) on your machine.)

For a complete listing of targets and options refer to the help:

```
$ python scons.py -h
```

If you don't have a recent version of Cython installed, you must install it or the `pylowl` steps above will fail.  To install a known compatible version of Cython in your home directory, do the following:

```
$ wget http://cython.org/release/Cython-0.19.2.tar.gz
$ tar -xzf Cython-0.19.2.tar.gz
$ cd Cython-0.19.2
$ python setup.py install --user
```
