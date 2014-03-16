# vim: ft=python

import os
import subprocess

env = Environment()

def py_ext_generator(source, target, env, for_signature):
    source_dir = os.path.split(env.GetBuildPath(source[0]))[0]
    target_dir = os.path.split(env.GetBuildPath(target[0]))[0]

    setup_path = os.path.join(source_dir, 'setup.py')
    Depends(target, setup_path)

    if os.path.isabs(source_dir):
        rel_setup_path = setup_path
    else:
        rel_setup_path = os.path.relpath(setup_path, start=target_dir)

    def build_py_ext(source, target, env):
        args = ['python', rel_setup_path, 'build_ext', '--inplace']
        print('%s$ %s' % (target_dir, ' '.join(args)))
        p = subprocess.Popen(args, cwd=target_dir)
        p.wait()
        return p.returncode

    return build_py_ext

py_ext = Builder(generator=py_ext_generator,
    suffix='$SHLIBSUFFIX', src_suffix='.pyx')

env.Append(BUILDERS = {'PyExt' : py_ext})

Export('env')

env.SConscript('src/SConscript', variant_dir='build', duplicate=0)
