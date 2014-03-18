# vim: ft=python

import os
import subprocess

env = Environment(ENV = os.environ)


def py_ext_generator(source, target, env, for_signature):
    source_dir = os.path.split(env.GetBuildPath(source[0]))[0]
    target_dir = os.path.split(env.GetBuildPath(target[0]))[0]

    setup_path = os.path.join(source_dir, 'setup.py')
    Depends(target, setup_path)

    def build_py_ext(source, target, env):
        args = 'python setup.py build_ext --inplace'.split()
        print('%s$ %s' % (target_dir, ' '.join(args)))
        p = subprocess.Popen(args, cwd=target_dir, env=env['ENV'])
        p.wait()
        return p.returncode

    return build_py_ext

py_ext = Builder(generator=py_ext_generator,
    suffix='$SHLIBSUFFIX', src_suffix='.pyx')

env.Append(BUILDERS = {'PyExt' : py_ext})


def untargeted_local_runner(env, source, args, run_dir):
    target = 'PHONY'

    Depends(target, source)

    def run(source, target, env):
        print('%s$ %s' % (run_dir, ' '.join(args)))
        p = subprocess.Popen(args, cwd=run_dir, env=env['ENV'])
        p.wait()
        return p.returncode

    pseudo = env.Pseudo(env.Command(target='PHONY', source=source, action=run))
    env.AlwaysBuild(pseudo)
    return pseudo

env.AddMethod(untargeted_local_runner, 'UntargetedLocalRunner')


Export('env')

env.SConscript('src/SConscript', variant_dir='build')
