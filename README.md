LittleOwl
=========

LittleOwl is a collection of software tools for fast randomized algorithms and applications thereof.  Hoot.

LittleOwl consists of two main components, the `lowl` and `pylowl` libraries, and several sub-projects in `pylowl/proj`, including a particle filter for streaming learning in LDA and an implementation of online variational inference for nHDP.  The `lowl` library (or `liblowl`) is a lightweight C library implementing an assortment of randomized algorithms; `pylowl` is a Python extension module that wraps `liblowl` via Cython.

System Requirements
-------------------

The `lowl` library itself has minimal dependencies.  It is known to build on modern Linux and OS X with several versions of GCC 4.x (e.g., 4.4, 4.8).

The Python extension module `pylowl` requires modern Python 2.x (e.g., 2.6, 2.7) and Cython (e.g., 0.19).  The build is known to fail on Cython 0.11.

The projects under `proj` have varying dependencies; however, most of them depend at least on `pylowl`, and as such require at minimum a modern Python 2.x (e.g., 2.6, 2.7).  (GCC and Cython are not necessarily required if you can convince someone to make a binary distribution for you.)  Other dependencies you can expect to see are `numpy` and `concrete`.

Building and Installing
-----------------------

We use `distutils` for general builds.  However, a `Makefile` is provided for standalone compilation of `liblowl`, if desired.  From the top of the repository, execute `make` to build `liblowl`.  Execute `make help` for more information.

In general, you'll want to use `distutils` to build `pylowl` and zero or more projects under `pylowl/proj`.  To build `pylowl`:

```
$ python setup.py build
```

To build `pylowl` with the particle filter for LDA:

```
$ python setup.py build --with-proj-pflda
```

Now, to run the particle filter (after you have preprocessed the 20 newsgroups dataset), add the build directory to your Python path and execute `pflda_run_pf`.  Note that the specific paths below may vary according to your platform:

```
$ export PYTHONPATH="build/lib.linux-x86_64-2.7:$PYTHONPATH"
$ build/scripts-2.7/pflda_run_pf /path/to/the/preprocessed/tng/dataset tng
```

To install `pylowl` to your user directory (generally `$HOME/.local` on Linux) with the batch Gibbs and particle filter learners for LDA:

```
$ python setup.py install --with-proj-bglda --with-proj-pflda --user
```

For help, pass the `--help` flag to `setup.py`, e.g.:

```
$ python setup.py --help
$ python setup.py --help build
$ python setup.py --help install
```

Finally, if you don't have a recent version of Cython installed you can build `liblowl` using the standalone `Makefile` but virtually every `distutils` build target will fail.  To install a known compatible version of Cython in your home directory, do the following:

```
$ wget http://cython.org/release/Cython-0.19.2.tar.gz
$ tar -xzf Cython-0.19.2.tar.gz
$ cd Cython-0.19.2
$ python setup.py install --user
```
