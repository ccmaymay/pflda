from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("lda", ["lda.pyx"],
        include_dirs=[numpy.get_include(), '../../pylowl', '../../lowl'],
        library_dirs=['../../pylowl', '../../lowl'],
    ),
]

def main():
    setup(
        name='pylowl_lda',
        version='0.0',
        description='Particle Filter for LDA',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules
    )

if __name__ == '__main__':
    main()
