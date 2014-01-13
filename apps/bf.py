import pylowl

def main(data_filename, io_flag, bf_filename):
    if io_flag not in ('-i', '-o'):
        raise Exception('Invalid io flag')

    print('Constructing bloom filter...')
    bf = pylowl.BloomFilter()
    print('Initializing bloom filter...')
    bf.init(1024 * 1024, 32)
    if io_flag == '-i':
        print('Reading bloom filter...')
        bf.read(bf_filename)
    else:
        print('Inserting data into bloom filter...')
        ngrams = dict()
        with open(data_filename) as f:
            for line in f:
                ngram = ' '.join(line.split()[:-1])
                ngrams[ngram] = len(ngrams)
                bf.insert(ngrams[ngram])
    print('Printing bloom filter...')
    bf.cPrint()
    if io_flag == '-o':
        print('Writing bloom filter...')
        bf.write(bf_filename)
    print('Leaving scope...')

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
