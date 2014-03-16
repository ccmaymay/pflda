from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("pylowl", ["pylowl.pyx"],
        libraries=["lowl"],
        include_dirs=['../lowl'],
        library_dirs=['../lowl'],
    ),
]

def main():
    setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)

if __name__ == '__main__':
    main()
