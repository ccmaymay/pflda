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
        with open(data_filename) as f:
            for line in f:
                ngram = ' '.join(line.split()[:-1])
                bf.insert(ngram)
        print('Writing bloom filter...')
        bf.write(bf_filename)
        ## bvd: the following spews bits to STDOUT
        #print('Printing bloom filter...')
        #bf.cPrint()
    else:
        bf_filename = args[0]
        ngram = args[1]
        print('Reading bloom filter...')
        bf.read(bf_filename)
        print('Querying bloom filter...')
        print(bf.query(ngram))
    print('Leaving scope...')

if __name__ == '__main__':
    main(*sys.argv[1:])
