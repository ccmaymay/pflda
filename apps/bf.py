import pylowl

def main(cmd, *args):
    if cmd not in ('read', 'write'):
        raise Exception('Invalid io flag')

    print('Constructing bloom filter...')
    bf = pylowl.BloomFilter()
    print('Initializing bloom filter...')
    bf.init(1024 * 1024, 32)
    if cmd == 'read':
        bf_filename = args[0]
        print('Reading bloom filter...')
        bf.read(bf_filename)
    else:
        data_filename = args[0]
        bf_filename = args[1]
        print('Inserting data into bloom filter...')
        ngrams = dict()
        with open(data_filename) as f:
            for line in f:
                ngram = ' '.join(line.split()[:-1])
                ngrams[ngram] = len(ngrams)
                bf.insert(ngrams[ngram])
        print('Writing bloom filter...')
        bf.write(bf_filename)
    #print('Printing bloom filter...')
    #bf.cPrint()
    print('Leaving scope...')

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
