import pylowl
import sys

def main(cmd, *args):
    if cmd not in ('read', 'query'):
        raise Exception('Invalid io flag')

    print('Constructing bloom filter...')
    bf = pylowl.BloomFilter()
    print('Initializing bloom filter...')
    bf.init(1024 * 1024, 32)
    if cmd == 'read':
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
        ## bvd: the following spews bits to STDOUT
        #print('Printing bloom filter...')
        #bf.cPrint()
    else:
        data_filename = args[0]
        bf_filename = args[1]
        query = ' '.join(args[2:])
        print('Loading dictionary...')
        ngrams = dict()
        with open(data_filename) as f:
            for line in f:
                ngram = ' '.join(line.split()[:-1])
                ngrams[ngram] = len(ngrams)
        print('Reading bloom filter...')
        bf.read(bf_filename)
        print('Querying bloom filter...')
        if query not in ngrams:
            ngrams[query] = len(ngrams)
        key = ngrams[query]
        print(bf.query(key))
    print('Leaving scope...')

if __name__ == '__main__':
    main(*sys.argv[1:])
