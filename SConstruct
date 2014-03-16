# vim: ft=python

import os
import subprocess

env = Environment()

def py_ext_generator(source, target, env, for_signature):
    (source_dir, source_filename) = os.path.split(env.GetBuildPath(source[0]))
    Depends(target, os.path.join(source_dir, 'setup.py'))

    def _action(source, target, env):
        p = subprocess.Popen(['python', 'setup.py', 'build_ext', '--inplace'],
            cwd=source_dir)
        p.wait()
        return p.returncode

    return _action

py_ext = Builder(generator=py_ext_generator,
    suffix='$SHLIBSUFFIX', src_suffix='.pyx')

env.Append(BUILDERS = {'PyExt' : py_ext})

Export('env')

env.SConscript('src/SConscript', variant_dir='build') #, duplicate=0)
