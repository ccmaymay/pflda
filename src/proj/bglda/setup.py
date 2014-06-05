from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension('bglda.core', ['bglda/core.pyx'],
        library_dirs=['../../lowl'],
        include_dirs=['../../lowl', numpy.get_include()],
    ),
]

def main():
    setup(
        name='bglda',
        version='0.1',
        description='Batch Gibbs Sampler for LDA',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules,
        py_modules=['bglda.data'],
        scripts=['bglda_run'],
    )

if __name__ == '__main__':
    main()
