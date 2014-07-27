from distutils.extension import Extension

def select(packages, package_data, ext_modules, scripts):
    packages.append('pylowl.proj.brightside')
    #ext_modules.append(
    #    Extension('pylowl.proj.brightside.core',
    #        ['src/pylowl/proj/brightside/core.pyx'],
    #        include_dirs=['src', 'src/lowl', numpy.get_include()],
    #        extra_compile_args=['-std=gnu99', '-O0', '-g'],
    #    ))
    packages.append('pylowl.proj.brightside.preproc')
    packages.append('pylowl.proj.brightside.postproc')
    package_data['pylowl.proj.brightside.postproc'] = ['*.html', '*.js']

    packages.append('pylowl.proj.brightside.hdp')
    packages.append('pylowl.proj.brightside.hdp.postproc')
    package_data['pylowl.proj.brightside.hdp.postproc'] = ['*.html', '*.js']

    packages.append('pylowl.proj.brightside.shdp')
    packages.append('pylowl.proj.brightside.shdp.postproc')
    package_data['pylowl.proj.brightside.shdp.postproc'] = ['*.html', '*.js']

    packages.append('pylowl.proj.brightside.m0')
    packages.append('pylowl.proj.brightside.m0.postproc')
    package_data['pylowl.proj.brightside.m0.postproc'] = ['*.html', '*.js']

    packages.append('pylowl.proj.brightside.m1')
    packages.append('pylowl.proj.brightside.m1.postproc')
    package_data['pylowl.proj.brightside.m1.postproc'] = ['*.html', '*.js']
