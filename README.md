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

We use `make` as a build system.  Note that `make` must always be run from the top-level directory.  To build and test `lowl` and `pylowl`:

```
$ make lowl-tests pylowl-tests
```

To install `lowl` locally:

```
$ make lowl-install INSTALL_PREFIX=$HOME
```

The `INSTALL_USER` flag is identical for the `lowl-install` target:

```
$ make lowl-install INSTALL_USER=1
```

To install `pylowl` to your home directory:

```
$ make pylowl-install INSTALL_PREFIX=$HOME
```

To install `pylowl` in the user installation scheme (location in home directory that is automatically in your Python module search path):

```
$ make pylowl-install INSTALL_USER=1
```

To build the Python extension module implementing a particle filter for LDA:

```
$ make pflda
```

To run the particle filter from the build directory, after you have imported the twenty newsgroups dataset (see the documentation in `src/proj/pflda` for details):

```
$ make pflda-fat
$ pushd build/proj/pflda
$ ./pflda_run_pf /path/to/20-newsgroups/dataset tng
$ popd
```

The `pflda-fat` target depends on `pflda` and also copies over the `lowl` and `pylowl` to facilitate running directly from `build/proj/pflda`.

The installation targets are decoupled; they do not depend on one another.  Thus, to run `pflda` from outside the build directory you must explicitly install `lowl`, `pylowl`, and `pflda`.  E.g.:

```
$ make lowl-install pylowl-install pflda-install INSTALL_USER=1
$ pushd /place/where/you/like/to/run/your/things
$ pflda_run_pf /path/to/20-newsgroups/dataset tng
$ popd
```

This assumes you have added the local Python library path and executable path to your environment.  The library path is likely searched by default but the executable path (where `pflda_run_pf` was installed) is likely not.

For a complete listing of targets and options refer to the help:

```
$ make help
```

Finally, if you don't have a recent version of Cython installed the `pylowl` targets above (and other targets that depend on them) will fail.  To install a known compatible version of Cython in your home directory, do the following:

```
$ wget http://cython.org/release/Cython-0.19.2.tar.gz
$ tar -xzf Cython-0.19.2.tar.gz
$ cd Cython-0.19.2
$ python setup.py install --user
```
