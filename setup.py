from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from glob import glob
import numpy

from distutils.command.build import build


packages=['pylowl', 'pylowl.proj']

ext_modules = [
    Extension('pylowl.core', ['src/pylowl/core.pyx']
            + [p for p in glob('src/lowl/*.c') if not p.endswith('/tests.c')],
        include_dirs=['src/pylowl', 'src/lowl'],
    ),
]

scripts = []


class selective_build(build):
    user_options = [
        ('with-bglda', None, 'Build bglda.'),
        ('with-pflda', None, 'Build pflda.'),
    ]

    def initialize_options(self):
        self.with_bglda = False
        self.with_pflda = False
        build.initialize_options(self)

    def run(self):
        if self.with_bglda:
            packages.append('pylowl.proj.bglda')
            ext_modules.append(
                Extension('pylowl.proj.bglda.core',
                    ['src/pylowl/proj/bglda/core.pyx'],
                    include_dirs=['src/pylowl', 'src/lowl', numpy.get_include()],
                ))
            scripts.append('src/pylowl/proj/bglda/bglda_run')

        if self.with_pflda:
            packages.append('pylowl.proj.pflda')
            ext_modules.append(
                Extension('pylowl.proj.pflda.core',
                    ['src/pylowl/proj/pflda/core.pyx'],
                    include_dirs=['src/pylowl', 'src/lowl', numpy.get_include()],
                ))
            scripts.append('src/pylowl/proj/pflda/pflda_run_pf')
            scripts.append('src/pylowl/proj/pflda/pflda_run_gibbs')

        build.run(self)

def main():
    setup(
        name='pylowl',
        version='0.0',
        description='Randomized Algorithms Library',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext, 'build': selective_build},
        package_dir={'': 'src'},
        packages=packages,
        ext_modules=ext_modules,
        scripts=scripts,
    )

if __name__ == '__main__':
    main()
