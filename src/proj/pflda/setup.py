from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension('pflda.core', ['pflda/core.pyx'],
        include_dirs=[numpy.get_include()],
    ),
]

def main():
    setup(
        name='pflda',
        version='0.1',
        description='Particle Filter for LDA',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules,
        py_modules=['pflda.data'],
        scripts=['pflda_run_pf', 'pflda_run_gibbs'],
    )

if __name__ == '__main__':
    main()
