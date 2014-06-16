from utils import *
from nose.tools import raises


def get_path_suffix_1_test():
    assert get_path_suffix('a', 'a') == '.'

def get_path_suffix_2_test():
    assert get_path_suffix('', '') == '.'

def get_path_suffix_3_test():
    assert get_path_suffix('a', '') == 'a'

@raises(Exception)
def get_path_suffix_4_test():
    get_path_suffix('a', 'b')

def get_path_suffix_5_test():
    assert get_path_suffix('a/b/c/d', 'a/b') == 'c/d'

def get_path_suffix_6_test():
    assert get_path_suffix('a/../a/./b/c/d/.', 'a/b') == 'c/d'

def get_path_suffix_7_test():
    assert get_path_suffix('a/b/c/d', 'a/../a/./b/.') == 'c/d'
