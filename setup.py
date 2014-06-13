from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from glob import glob
import numpy

from distutils.command.build import build
from distutils.command.install import install


packages=['pylowl', 'pylowl.proj']

ext_modules = [
    Extension('pylowl.core', ['src/pylowl/core.pyx']
            + [p for p in glob('src/lowl/*.c') if not p.endswith('/tests.c')],
        include_dirs=['src/pylowl', 'src/lowl'],
    ),
]

scripts = []

selected_proj = set()


def select_proj_bglda():
    packages.append('pylowl.proj.bglda')
    ext_modules.append(
        Extension('pylowl.proj.bglda.core',
            ['src/pylowl/proj/bglda/core.pyx'],
            include_dirs=['src/pylowl', 'src/lowl', numpy.get_include()],
        ))
    scripts.append('src/pylowl/proj/bglda/bglda_run')

def select_proj_pflda():
    packages.append('pylowl.proj.pflda')
    ext_modules.append(
        Extension('pylowl.proj.pflda.core',
            ['src/pylowl/proj/pflda/core.pyx'],
            include_dirs=['src/pylowl', 'src/lowl', numpy.get_include()],
        ))
    scripts.append('src/pylowl/proj/pflda/pflda_run_gibbs', 'src/pylowl/proj/pflda/pflda_run_pf')


k = None
PROJECTS = [k[len('select_proj_'):] for k in globals() if k.startswith('select_proj_')]

USER_OPTIONS = [('with-proj-' + p, None, 'Install ' + p) for p in PROJECTS]


def make_initialize_options(superclass):
    def initialize_options(o):
        for p in PROJECTS:
            attr_name = 'with_proj_' + p
            setattr(o, attr_name, False)
        superclass.initialize_options(o)
    return initialize_options


def make_run(superclass):
    def run(o):
        for p in PROJECTS:
            attr_name = 'with_proj_' + p
            if getattr(o, attr_name) and p not in selected_proj:
                selector_name = 'select_proj_' + p
                selector = globals()[selector_name]
                selector()
                selected_proj.add(p)
        superclass.run(o)
    return run


class selective_build(build):
    user_options = build.user_options + USER_OPTIONS
    initialize_options = make_initialize_options(build)
    run = make_run(build)


class selective_install(install):
    user_options = install.user_options + USER_OPTIONS
    initialize_options = make_initialize_options(install)
    run = make_run(install)


def main():
    setup(
        name='pylowl',
        version='0.0',
        description='Randomized Algorithms Library',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext, 'build': selective_build, 'install': selective_install},
        package_dir={'': 'src'},
        packages=packages,
        ext_modules=ext_modules,
        scripts=scripts,
    )


if __name__ == '__main__':
    main()
