import pylowl

def main(filename):
    print('Constructing bloom filter...')
    bf = pylowl.BloomFilter()
    print('Initializing bloom filter...')
    bf.init(1024, 32)
    print('Inserting data into bloom filter...')
    ngrams = dict()
    with open(filename) as f:
        for line in f:
            ngram = ' '.join(line.split()[:-1])
            ngrams[ngram] = len(ngrams)
            bf.insert(ngrams[ngram])
    print('Printing bloom filter...')
    bf.cPrint()
    print('Leaving scope...')

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
