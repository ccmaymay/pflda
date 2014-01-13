from pylowl import Table
from random import choice, randint
from string import lowercase
import gc

MAX_STRING_LEN = 42
NUM_ENTRIES = 16

def random_string(max_len):
    return ''.join([choice(lowercase) for x in range(randint(0, max_len))])

def insert_string(t, store):
    s = random_string(MAX_STRING_LEN)
    store.append(s) # NOTE need this or strings will be gc'd
    i = t.insert(s)
    print '*' * 80
    print 'Insert: %d: %s' % (i, s)
    print t
    return i

t = Table()
indices = []
store = []
for i in range(NUM_ENTRIES):
    indices.append(insert_string(t, store))
print '*' * 80
print '*' * 80
print 'Inserted %d entries.' % NUM_ENTRIES

gc.collect()
print 'Garbage collected.'

for idx in indices:
    print 'Get: %d: %s' % (idx, t[idx])
