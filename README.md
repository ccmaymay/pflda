LittleOwl
=========

LittleOwl is a collection of software tools for fast randomized algorithms and applications thereof.  Hoot.

LittleOwl is separated into two sub-projects, `core` and `apps`.  `core` contains a lightweight C library for randomized algorithms, called `liblowl`.  It also contains a Python interface, `pylowl`, via Cython.  `apps` contains several Python applications that use `pylowl` to do nifty things.

System Requirements
-------------------

The LittleOwl core library `liblowl` has minimal dependencies.  It is known to build on modern Linux and OS X with several versions of GCC 4.x (e.g., 4.4, 4.8).

The Python bridge `pylowl` requires modern Python 2.x (e.g., 2.6, 2.7) and Cython (e.g., 0.19).  The build is known to fail on Cython 0.11.

The `apps` library requires modern Python 2.x (e.g., 2.6, 2.7).

Installing
----------

To build and test the `liblowl` library:

```
$ cd core
$ make test
```

Now, if you want to build the bridge:

```
$ make
```

If you don't have a recent version of Cython installed, you must install it or the step above will fail.  To install a known compatible version of Cython in your home directory, do the following:

```
$ wget http://cython.org/release/Cython-0.19.2.tar.gz
$ tar -xzf Cython-0.19.2.tar.gz
$ cd Cython-0.19.2
$ python setup.py install --user
```

To install the built libraries to `apps`, use the `install` target:

```
$ make install
```

To run Python demos using the `pylowl` library, use the `apps` target:

```
$ make apps
```

Note that `install` will automatically build `liblowl` if it does not exist, and `apps` will automatically invoke `install` to copy `pylowl` to `apps`.  Hence, if you're in a hurry, just run `make apps` to do everything in one fell swoop.
