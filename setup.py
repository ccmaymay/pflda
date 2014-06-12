from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from glob import glob
import numpy

ext_modules = [
    Extension('pylowl.core', ['src/pylowl/core.pyx']
            + [p for p in glob('src/lowl/*.c') if not p.endswith('/tests.c')],
        include_dirs=['src/pylowl', 'src/lowl'],
    ),
    Extension('pylowl.proj.pflda.core', ['src/pylowl/proj/pflda/core.pyx'],
        include_dirs=['src/pylowl', 'src/lowl', numpy.get_include()],
    ),
    Extension('pylowl.proj.bglda.core', ['src/pylowl/proj/bglda/core.pyx'],
        include_dirs=['src/pylowl', 'src/lowl', numpy.get_include()],
    ),
]

def main():
    setup(
        name='pylowl',
        version='0.0',
        description='Randomized Algorithms Library',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext},
        package_dir={'': 'src'},
        packages=['pylowl', 'pylowl.proj', 'pylowl.proj.pflda', 'pylowl.proj.bglda'],
        ext_modules=ext_modules,
        scripts=['src/pylowl/proj/bglda/bglda_run', 'src/pylowl/proj/pflda/pflda_run_pf', 'src/pylowl/proj/pflda/pflda_run_gibbs'],
    )

if __name__ == '__main__':
    main()
