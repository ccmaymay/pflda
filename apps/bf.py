import pylowl

def main():
    print('Constructing bloom filter...')
    bf = pylowl.BloomFilter()
    print('Initializing bloom filter...')
    bf.init(1024, 64)
    print('Inserting data into bloom filter...')
    bf.insert(42)
    print('Printing bloom filter...')
    bf.cPrint()
    print('Leaving scope...')

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
