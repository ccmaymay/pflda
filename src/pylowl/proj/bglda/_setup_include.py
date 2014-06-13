from distutils.extension import Extension
import numpy

def select(packages, ext_modules, scripts):
    packages.append('pylowl.proj.bglda')
    ext_modules.append(
        Extension('pylowl.proj.bglda.core',
            ['src/pylowl/proj/bglda/core.pyx'],
            include_dirs=['src', 'src/lowl', numpy.get_include()],
        ))
    scripts.append('src/pylowl/proj/bglda/bglda_run')
