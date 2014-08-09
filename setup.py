from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


packages=['pflda']

package_data = {}

ext_modules = [
    Extension('pflda.core',
        ['src/pflda/core.pyx'],
        include_dirs=['src', numpy.get_include()],
        extra_compile_args=['-std=gnu99'],
    ),
]

scripts = [
    'src/pflda/pflda_run_gibbs',
    'src/pflda/pflda_run_pf',
]


def main():
    setup(
        author='hltcoe',
        author_email='hltcoe',
        name='pflda',
        version='0.1',
        description='Particle filter algorithm for LDA',
        url='https://gitlab.hltcoe.jhu.edu/littleowl/pflda',
        cmdclass={'build_ext': build_ext},
        package_dir={'': 'src'},
        packages=packages,
        package_data=package_data,
        ext_modules=ext_modules,
        scripts=scripts,
    )


if __name__ == '__main__':
    main()
