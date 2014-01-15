import pylowl
import sys
import time


def main(cmd, *args):
    if cmd not in ('read', 'sample'):
        raise Exception('Invalid io flag')

    pylowl.srandom(int(time.time()))

    print('Constructing reservoir sampler...')
    rs = pylowl.ValuedReservoirSampler()
    print('Initializing reservoir sampler...')
    rs.init(64)
    if cmd == 'read':
        data_filename = args[0]
        rs_filename = args[1]
        values_filename = args[2]
        print('Inserting data into reservoir sampler...')
        ngrams = dict()
        with open(data_filename) as f:
            for line in f:
                ngram = ' '.join(line.split()[:-1])
                if ngram not in ngrams:
                    ngrams[ngram] = len(ngrams)
                key = ngrams[ngram]
                (inserted, ejected_key) = rs.insert(key, ngram)
                if inserted:
                    if ejected_key is None:
                        print('+ "%s"' % ngram)
                    else:
                        print('+ "%s" - "%s"' % (ngram, ejected_key))
        print('Writing reservoir sampler...')
        rs.write(rs_filename, values_filename)
    else:
        rs_filename = args[0]
        values_filename = args[1]
        print('Reading reservoir sampler...')
        rs.read(rs_filename, values_filename)
        print('Sampling from reservoir...')
        #rs.cPrint()
        print(rs.sample())
        print(rs.sample())
        print(rs.sample())
    print('Leaving scope...')


if __name__ == '__main__':
    main(*sys.argv[1:])
