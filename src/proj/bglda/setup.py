from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("bglda.core", ["core.pyx"],
        include_dirs=[numpy.get_include(), '../../pylowl', '../../lowl'],
        library_dirs=['../../pylowl', '../../lowl'],
    ),
]

def main():
    setup(
        name='bglda',
        version='0.0',
        description='Batch Gibbs Sampler for LDA',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext},
        package_dir = {'bglda': ''},
        packages=['bglda'],
        ext_modules=ext_modules,
        py_modules=['bglda.data'],
    )

if __name__ == '__main__':
    main()
