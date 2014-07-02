import os
import sys
import logging
from importlib import import_module


def iter_test_funcs(module):
    for name in dir(module):
        obj = getattr(module, name)
        if not name.startswith('.') and name.endswith('_test') and callable(obj):
            yield obj


def iter_test_modules(src_dir, parent_module_prefix=''):
    for entry in os.listdir(src_dir):
        if not entry.startswith('.'):
            path = os.path.join(src_dir, entry)
            if os.path.isdir(path):
                for m in iter_test_modules(path, parent_module_prefix + entry + '.'):
                    yield m
            elif os.path.isfile(path):
                if entry.endswith('_test.py') or entry.endswith('_test.pyx'):
                    module_name = parent_module_prefix + entry[:entry.rfind('.')]
                    try:
                        yield import_module(module_name)
                    except:
                        pass


def run_tests(src_dirs):
    successes = 0
    errors = 0
    for src_dir in src_dirs:
        for m in iter_test_modules(src_dir):
            for f in iter_test_funcs(m):
                try:
                    f()
                except:
                    errors += 1
                    logging.error('', exc_info=True)
                else:
                    successes += 1
    return (successes, errors)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    (successes, errors) = run_tests(sys.argv[1:] if len(sys.argv) > 1 else ('.',))
    num_tests = errors + successes
    if errors == 0:
        if successes == 0:
            mood = 'O_o'
        else:
            mood = '^_^'
    elif errors/float(num_tests) < 0.3:
        mood = '-_-'
    elif errors < num_tests:
        mood = 't_t'
    else:
        mood = 'x_x'

    logging.info('%d/%d tests passed %s' % (successes, num_tests, mood))
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)
