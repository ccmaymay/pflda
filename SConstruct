# vim: ft=python

import os
import subprocess


AddOption('--prefix',
          dest='prefix',
          type='string',
          nargs=1,
          action='store',
          metavar='DIR',
          help='installation (prefix) dir')

PREFIX = GetOption('prefix')
if PREFIX:
    PREFIX = os.path.abspath(PREFIX)
else:
    PREFIX = '/'

env = Environment(ENV=os.environ, PREFIX=PREFIX)


def _str_add_ext(x, ext):
    '''
    If x is a string that does not end in ext then return x + ext,
    else return x.
    '''
    if isinstance(x, str):
        if x.endswith(ext):
            return x
        else:
            return x + ext
    else:
        return x


def _list_add_ext(x, ext):
    '''
    Return list whose elements are the elements of x with extension
    ext appended (where appropriate).
    '''
    return [_str_add_ext(elt, ext) for elt in x]


def _make_local_runner(args):
    run_dir = Dir('.').path
    def local_runner(target, source, env):
        print('%s$ %s' % (run_dir, ' '.join(args)))
        p = subprocess.Popen(args, cwd=run_dir, env=env['ENV'])
        p.wait()
        return p.returncode
    return local_runner


def distutils_setup(_env, _target, args, _source, pseudo=False):
    '''
    Run python setup.py in the current build dir with the provided args.
    If pseudo is true, also specify that target is phony and should
    always be built.
    '''

    # Specify dependency on setup.py manually;
    # other dependencies are implicitly specified in _source
    Depends(_target, 'setup.py')

    run_distutils_setup = _make_local_runner(['python', 'setup.py'] + args)

    ret = _env.Command(target=_target, source=_source,
        action=run_distutils_setup)
    if pseudo:
        return _env.Pseudo(_env.AlwaysBuild(ret))
    else:
        return ret

env.AddMethod(distutils_setup, 'DistutilsSetup')


def distutils_build_ext(env, target, source, deps=None):
    '''
    Run python setup.py build_ext --inplace in the current build dir
    Add extension .pyx to source (non-.pyx dependencies such as .pxd
    files should be specified in deps).  Add shared library extension
    (e.g., .so) to target.
    '''
    if deps is None:
        deps = []
    my_target = _list_add_ext(Flatten(target), env.subst('$SHLIBSUFFIX'))
    my_source = _list_add_ext(Flatten(source), '.pyx') + deps
    return env.DistutilsSetup(my_target, ['build_ext', '--inplace'], my_source)

env.AddMethod(distutils_build_ext, 'DistutilsBuildExt')


def distutils_install(env, target, source, deps=None, args=None):
    '''
    Run python setup.py install in the current build dir with the
    provided args.  Set phony target.  Add extension .pyx to source
    (non-.pyx dependencies such as .pxd files should be specified in
    deps).
    '''
    if deps is None:
        deps = []
    if args is None:
        args = []
    my_source = _list_add_ext(Flatten(source), '.pyx') + deps
    return env.DistutilsSetup(target, ['install'] + args, my_source,
        pseudo=True)

env.AddMethod(distutils_install, 'DistutilsInstall')


def no_target_local_runner(env, target, args, source):
    local_runner = _make_local_runner(args)
    ret = env.Alias(target=target, source=source, action=local_runner)
    return env.AlwaysBuild(ret)

env.AddMethod(no_target_local_runner, 'NoTargetLocalRunner')


Export('env')

env.SConscript('src/SConscript', variant_dir='build')
