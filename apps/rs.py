import pylowl
import sys
import time


def main(cmd, *args):
    if cmd not in ('read', 'sample'):
        raise Exception('Invalid io flag')

    pylowl.srandom(int(time.time()))

    print('Constructing reservoir sampler...')
    rs = pylowl.ValuedReservoirSampler(64)
    print('Initializing reservoir sampler...')
    if cmd == 'read':
        data_filename = args[0]
        rs_filename = args[1]
        values_filename = args[2]
        print('Inserting data into reservoir sampler...')
        with open(data_filename) as f:
            for line in f:
                ngram = ' '.join(line.split()[:-1])
                (inserted, idx, ejected, ejected_ngram) = rs.insert(ngram)
                if inserted:
                    if ejected_ngram is None:
                        print('+ "%s"' % ngram)
                    else:
                        print('+ "%s" - "%s"' % (ngram, ejected_ngram))
        print('Writing reservoir sampler...')
        rs.write(rs_filename, values_filename)
    else:
        rs_filename = args[0]
        values_filename = args[1]
        print('Reading reservoir sampler...')
        rs.read(rs_filename, values_filename)
        print('Printing reservoir...')
        for i in range(rs.occupied()):
            print(rs.get(i))
    print('Leaving scope...')


if __name__ == '__main__':
    main(*sys.argv[1:])
