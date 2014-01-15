import pylowl
import sys

def main(cmd, *args):
    if cmd not in ('read', 'print'):
        raise Exception('Invalid io flag')

    print('Constructing reservoir sampler...')
    rs = pylowl.ReservoirSampler()
    print('Initializing reservoir sampler...')
    rs.init(16)
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
                print (ngram, rs.insert(key, ngram))
        print('Writing reservoir sampler...')
        rs.write(rs_filename, values_filename)
    else:
        rs_filename = args[0]
        values_filename = args[1]
        print('Reading reservoir sampler...')
        rs.read(rs_filename, values_filename)
        print('Querying reservoir sampler...')
        rs.cPrint()
    print('Leaving scope...')

if __name__ == '__main__':
    main(*sys.argv[1:])
