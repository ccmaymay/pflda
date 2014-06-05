from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension('pylowl', ['pylowl.pyx'],
        libraries=['lowl'],
        library_dirs=['../lowl'],
        include_dirs=['../lowl'],
    ),
]

def main():
    setup(
        name='pylowl',
        version='0.0',
        description='Randomized Algorithms Library',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules
    )

if __name__ == '__main__':
    main()
