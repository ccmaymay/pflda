import pylowl

def main():
    print('Constructing bloom filter...')
    bf = pylowl.BloomFilter()
    print('Initializing bloom filter...')
    bf.init(32, 64)
    print('Printing bloom filter...')
    bf.cPrint()
    print('Leaving scope...')

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
