import os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from glob import glob
from importlib import import_module

from distutils.command.build import build
from distutils.command.install import install
from distutils.command.clean import clean
from distutils.command.sdist import sdist
from distutils.command.bdist import bdist


packages=['pylowl', 'pylowl.proj']

package_data = {}

ext_modules = [
    Extension('pylowl.core', ['src/pylowl/core.pyx']
            + [p for p in glob('src/lowl/*.c') if not p.endswith('/tests.c')],
        include_dirs=['src', 'src/lowl'],
        extra_compile_args=['-std=gnu99'],
    ),
]

scripts = []

selected_proj = set()

projects = []

for proj_dir in glob('src/pylowl/proj/*'):
    if os.path.isdir(proj_dir):
        include_file = os.path.join(proj_dir, '_setup_include.py')
        if os.path.isfile(include_file):
            proj_name = os.path.basename(proj_dir)
            proj_include_package_name = 'src.pylowl.proj.%s.%s' % (proj_name, '_setup_include')
            m = import_module(proj_include_package_name)
            projects.append((proj_name, m.select))


USER_OPTIONS = ([('with-proj-' + p[0],
                  None,
                  'Include ' + p[0] + ' project in build/install/etc.')
                 for p in projects]
                + [('with-all-proj',
                    None,
                    'Include all projects in build/install/etc.')])


def make_initialize_options(superclass):
    def initialize_options(o):
        o.with_all_proj = False
        for p in projects:
            attr_name = 'with_proj_' + p[0]
            setattr(o, attr_name, False)
        superclass.initialize_options(o)
    return initialize_options


def make_run(superclass):
    def run(o):
        for p in projects:
            attr_name = 'with_proj_' + p[0]
            selected = o.with_all_proj or getattr(o, attr_name)
            if selected and p[0] not in selected_proj:
                p[1](packages, package_data, ext_modules, scripts)
                selected_proj.add(p[0])
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


class selective_clean(clean):
    user_options = clean.user_options + USER_OPTIONS
    initialize_options = make_initialize_options(clean)
    run = make_run(clean)


class selective_sdist(sdist):
    user_options = sdist.user_options + USER_OPTIONS
    initialize_options = make_initialize_options(sdist)
    run = make_run(sdist)


class selective_bdist(bdist):
    user_options = bdist.user_options + USER_OPTIONS
    initialize_options = make_initialize_options(bdist)
    run = make_run(bdist)


def main():
    setup(
        author='hltcoe',
        author_email='hltcoe',
        name='pylowl',
        version='0.1',
        description='Randomized Algorithms Library',
        url='https://gitlab.hltcoe.jhu.edu/klevin/littleowl',
        cmdclass={'build_ext': build_ext, 'build': selective_build, 'install': selective_install, 'clean': selective_clean, 'sdist': selective_sdist, 'bdist': selective_bdist},
        package_dir={'': 'src'},
        packages=packages,
        ext_modules=ext_modules,
        scripts=scripts,
    )


if __name__ == '__main__':
    main()
