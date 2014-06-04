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

The `pylowl-install` target also accepts the `--prefix` flag; its semantics and those of `--user` are defined by `distutils`.

To build the Python extension module implementing a particle filter for LDA:

```
$ python scons.py pflda
```

To run the particle filter from the build directory, after you have imported the twenty newsgroups dataset (see the documentation in `src/proj/pflda` for details):

```
$ python scons.py pflda-fat
$ pushd build/proj/pflda
$ ./pflda_run_pf /path/to/20-newsgroups/dataset tng
$ popd
```

The `pflda-fat` target depends on `pflda` and also copies over the `lowl` and `pylowl` to facilitate running directly from `build/proj/pflda`.

The installation targets are decoupled; they do not depend on one another.  Thus, to run `pflda` from outside the build directory you must explicitly install `lowl`, `pylowl`, and `pflda`.  E.g.:

```
$ python scons.py --user lowl-install pylowl-install pflda-install
$ pushd /place/where/you/like/to/run/your/things
$ pflda_run_pf /path/to/20-newsgroups/dataset
$ popd
```

This assumes you have added the local Python library path and executable path to your environment.  The library path is likely searched by default but the executable path (where `pflda_run_pf` was installed) is likely not.

Scons expects to be run from the root of the repository; however, you may run from a subdirectory by passing the `-u` flag to scons.  E.g., building `pflda` from the `src/proj/pflda` subdirectory:

```
$ pushd src/proj/pflda
$ python ../../../scons.py -u pflda
$ popd
```

This makes more sense if you have [installed scons](http://scons.org/download.php) on your machine, in which case you can run scons simply as `scons`.

For a complete listing of targets and options refer to the help:

```
$ python scons.py -h
```

Finally, if you don't have a recent version of Cython installed the `pylowl` targets above (and other targets that depend on them) will fail.  To install a known compatible version of Cython in your home directory, do the following:

```
$ wget http://cython.org/release/Cython-0.19.2.tar.gz
$ tar -xzf Cython-0.19.2.tar.gz
$ cd Cython-0.19.2
$ python setup.py install --user
```
