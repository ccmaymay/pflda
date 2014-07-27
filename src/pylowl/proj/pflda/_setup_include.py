from distutils.extension import Extension
import numpy

def select(packages, package_data, ext_modules, scripts):
    packages.append('pylowl.proj.pflda')
    ext_modules.append(
        Extension('pylowl.proj.pflda.core',
            ['src/pylowl/proj/pflda/core.pyx'],
            include_dirs=['src', 'src/lowl', numpy.get_include()],
            extra_compile_args=['-std=gnu99'],
        ))
    scripts.append('src/pylowl/proj/pflda/pflda_run_gibbs')
    scripts.append('src/pylowl/proj/pflda/pflda_run_pf')
