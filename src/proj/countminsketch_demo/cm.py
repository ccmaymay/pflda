import pylowl
import sys
import time


def main(cmd, *args):
    if cmd not in ('read', 'query'):
        raise Exception('Invalid io flag')

    pylowl.srandom(int(time.time()))

    print('Constructing count-min sketch...')
    cm = pylowl.CountMinSketch()
    print('Initializing count-min sketch...')
    cm.init(1024 * 1024, 32)
    if cmd == 'read':
        data_filename = args[0]
        cm_filename = args[1]
        print('Inserting data into count-min sketch...')
        with open(data_filename) as f:
            for line in f:
                pieces = line.split()
                ngram = ' '.join(pieces[:-1])
                count = int(pieces[-1])
                cm.add(ngram, len(ngram), count)
        print('Writing count-min sketch...')
        cm.write(cm_filename)
    else:
        cm_filename = args[0]
        ngram = args[1]
        print('Reading count-min sketch...')
        cm.read(cm_filename)
        print('Querying count-min sketch...')
        print(cm.query(ngram, len(ngram)))
    print('Leaving scope...')

if __name__ == '__main__':
    main(*sys.argv[1:])
