# vim: ft=python


import os
import platform
import subprocess


install_prefix_help = 'install prefix dir'
AddOption('--prefix',
          dest='install_prefix',
          type='string',
          nargs=1,
          action='store',
          metavar='DIR',
          help=install_prefix_help)

install_user_help = 'set user install scheme for Python and set prefix dir to home (overrides prefix)'
AddOption('--user',
          dest='install_user',
          action='store_true',
          help=install_user_help)

debug_help = 'compile with debugging symbols'
AddOption('--dbg',
          dest='debug',
          action='store_true',
          help=debug_help)

Help('''
Usage:
    scons [option ...] [target ...]

    Options:
        --prefix  %(install_prefix_help)s
        --user    %(install_user_help)s
        --dbg     %(debug_help)s

    Targets:
''' % locals())


INSTALL_USER = GetOption('install_user')

INSTALL_PREFIX = GetOption('install_prefix')
if INSTALL_USER:
    INSTALL_PREFIX = os.path.expanduser('~')
elif INSTALL_PREFIX:
    INSTALL_PREFIX = os.path.abspath(INSTALL_PREFIX)

DEBUG = GetOption('debug')


env = Environment(ENV=os.environ,
    INSTALL_PREFIX=INSTALL_PREFIX,
    INSTALL_USER=INSTALL_USER)
env.Append(CCFLAGS=['-std=gnu99'])
if DEBUG:
    env.Append(CCFLAGS=['-O0', '-g', '-Wall', '-Wextra'])
else:
    env.Append(CCFLAGS=['-O2'])

if platform.system() == 'Darwin':
    env['LD_LIB_PATH_ENV_VAR'] = 'DYLD_LIBRARY_PATH'
    # (TODO) temporary hack because OS X is dumb
    if 'ARCHFLAGS' in env['ENV']:
        env['ENV']['ARCHFLAGS'] += ' -Wno-error=unused-command-line-argument-hard-error-in-future'
    else:
        env['ENV']['ARCHFLAGS'] = '-Wno-error=unused-command-line-argument-hard-error-in-future'
else:
    env['LD_LIB_PATH_ENV_VAR'] = 'LD_LIBRARY_PATH'

env['PY_PATH_ENV_VAR'] = 'PYTHONPATH'


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
    '''
    Return action (function) that runs specified command in current
    build dir.
    '''
    run_dir = Dir('.').path
    def local_runner(target, source, env):
        print('%s$ %s' % (run_dir, ' '.join(args)))
        p = subprocess.Popen(args, cwd=run_dir, env=env['ENV'])
        p.wait()
        return p.returncode
    return local_runner


def distutils_build_ext(env, target, source, deps=None, args=None):
    '''
    Run python setup.py build_ext --inplace in the current build dir
    with provided args.
    Add extension .pyx to source (non-.pyx dependencies such as .pxd
    files should be specified in deps).  Add shared library extension
    (e.g., .so) to target.
    '''
    if deps is None:
        deps = []
    if args is None:
        args = []
    my_target = _list_add_ext(Flatten(target), env.subst('$SHLIBSUFFIX'))
    my_source = _list_add_ext(Flatten(source), '.pyx') + deps + ['setup.py']
    full_args = 'python setup.py build_ext --inplace'.split() + args
    local_runner = _make_local_runner(full_args)
    return env.Command(target=my_target, source=my_source, action=local_runner)

env.AddMethod(distutils_build_ext, 'DistutilsBuildExt')


def distutils_install(env, alias, source, deps=None, args=None):
    '''
    Run python setup.py install in the current build dir with the
    provided args.  Add extension .pyx to source
    (non-.pyx dependencies such as .pxd files should be specified in
    deps).  Set specified alias to command.
    '''
    if deps is None:
        deps = []
    if args is None:
        args = []

    if INSTALL_USER:
        args.append('--user')
    elif INSTALL_PREFIX:
        args.append(env.subst('--prefix=$INSTALL_PREFIX'))

    my_source = (_list_add_ext(Flatten(source), env.subst('$SHLIBSUFFIX'))
        + deps + ['setup.py'])
    return env.NoTargetLocalRunner(alias,
        'python setup.py install'.split() + args,
        my_source)

env.AddMethod(distutils_install, 'DistutilsInstall')


def no_target_local_runner(env, alias, args, source):
    '''
    Run specified command (that does not produce any output) from the
    current build dir.  Set specified alias to command.
    '''
    local_runner = _make_local_runner(args)
    ret = env.Alias(target=alias, source=source, action=local_runner)
    return env.AlwaysBuild(ret)

env.AddMethod(no_target_local_runner, 'NoTargetLocalRunner')


Export('env')

env.SConscript('src/SConscript', variant_dir='build')
