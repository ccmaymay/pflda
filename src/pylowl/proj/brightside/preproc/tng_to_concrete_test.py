from pylowl.proj.brightside.preproc.tng_to_concrete import *


def contains_email_1_test():
    assert not contains_email('')

def contains_email_2_test():
    assert not contains_email('the')

def contains_email_3_test():
    assert not contains_email('filter_raw.py')

def contains_email_4_test():
    assert not contains_email('foo@local')

def contains_email_5_test():
    assert not contains_email('@twitter')

def contains_email_6_test():
    assert contains_email('example.123@example.com.au')

def contains_email_7_test():
    assert contains_email('@@@a@@@b@@@c...d...e...')
