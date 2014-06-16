from distutils.extension import Extension

def select(packages, ext_modules, scripts):
    packages.append('pylowl.proj.brightside')
    packages.append('pylowl.proj.brightside.preproc')
    packages.append('pylowl.proj.brightside.postproc')
