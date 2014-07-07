from utils import *


def raises(f, ex_type):
    '''
    Return true iff f, when called, raises an exception of type ex_type
    '''

    try:
        f()
    except ex_type:
        return True
    except:
        return False
    else:
        return False


def get_path_suffix_1_test():
    assert get_path_suffix('a', 'a') == '.'

def get_path_suffix_2_test():
    assert get_path_suffix('', '') == '.'

def get_path_suffix_3_test():
    assert get_path_suffix('a', '') == 'a'

def get_path_suffix_4_test():
    raises(lambda: get_path_suffix('a', 'b'), Exception)

def get_path_suffix_5_test():
    assert get_path_suffix('a/b/c/d', 'a/b') == 'c/d'

def get_path_suffix_6_test():
    assert get_path_suffix('a/../a/./b/c/d/.', 'a/b') == 'c/d'

def get_path_suffix_7_test():
    assert get_path_suffix('a/b/c/d', 'a/../a/./b/.') == 'c/d'
