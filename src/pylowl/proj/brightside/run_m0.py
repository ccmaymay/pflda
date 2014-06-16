import logging
import time
import os
import sys
from corpus import Corpus
from utils import take
import m0
import cPickle
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob


LOG_BASENAME = 'log'
OPTIONS_BASENAME = 'options.dat'
DEFAULT_OPTIONS = dict(
    log_level='INFO',
    trunc='1,20,10,5',
    D=None,
    W=None,
    lambda0=0.01,
    beta=1.0,
    alpha=1.0,
    gamma1=1.0,
    gamma2=1.0,
    kappa=0.5,
    iota=1.0,
    delta=1e-3,
    omicron=None,
    xi=0.5,
    batchsize=100,
    max_iter=None,
    max_time=None,
    var_converge=0.0001,
    random_seed=None,
    data_path=None,
    test_data_path=None,
    output_dir='output',
    test_samples=None,
    test_train_frac=0.9,
    save_lag=500,
    pass_ratio=0.5,
    scale=1.0,
    adding_noise=False,
    streaming=False,
    fixed_lag=False,
    save_model=False,
    init_samples=None,
)


def main(argv=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**DEFAULT_OPTIONS)

    parser.add_argument("--log_level", type=str,
                      choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                      help="log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--trunc", type=str,
                      help="comma-separated list of truncations (per level)")
    parser.add_argument("--D", type=int,
                      help="number of documents (None: auto)")
    parser.add_argument("--W", type=int,
                      help="size of vocabulary (None: auto)")
    parser.add_argument("--lambda0", type=float,
                      help="the topic Dirichlet")
    parser.add_argument("--beta", type=float,
                      help="beta value")
    parser.add_argument("--alpha", type=float,
                      help="alpha value")
    parser.add_argument("--gamma1", type=float,
                      help="gamma1 value")
    parser.add_argument("--gamma2", type=float,
                      help="gamma2 value")
    parser.add_argument("--kappa", type=float,
                      help="learning rate")
    parser.add_argument("--iota", type=float,
                      help="slow down")
    parser.add_argument("--delta", type=float,
                      help="greedy subtree selection stopping crit")
    parser.add_argument("--omicron", type=float,
                      help="effective no. documents in initialization (None: actual)")
    parser.add_argument("--xi", type=float,
                      help="fraction of topic mass derived from data in initialization")
    parser.add_argument("--batchsize", type=int,
                      help="batch size")
    parser.add_argument("--max_iter", type=int,
                      help="max iterations for training (None: no max)")
    parser.add_argument("--max_time", type=int,
                      help="max time in seconds for training (None: no max)")
    parser.add_argument("--var_converge", type=float,
                      help="relative change on doc lower bound")
    parser.add_argument("--random_seed", type=int,
                      help="the random seed (None: auto)")
    parser.add_argument("--data_path", type=str,
                      help="training data path or glob pattern")
    parser.add_argument("--test_data_path", type=str,
                      help="testing data path")
    parser.add_argument("--test_train_frac", type=float,
                      help="fraction of testing docs on which to infer local distributions")
    parser.add_argument("--output_dir", type=str,
                      help="output directory")
    parser.add_argument("--save_lag", type=int,
                      help="the minimal saving lag, increasing as save_lag * 2^i, with max i as 10; default 500.")
    parser.add_argument("--init_samples", type=int,
                      help="number of initialization documents (nested k-means init)")
    parser.add_argument("--test_samples", type=int,
                      help="number of test documents (None: auto)")
    parser.add_argument("--scale", type=float,
                      help="scaling parameter for learning rate")
    parser.add_argument("--adding_noise", action="store_true",
                      help="add noise to the first couple of iterations")
    parser.add_argument("--streaming", action="store_true",
                      help="process data in streaming fashion (D and W must be specified)")
    parser.add_argument("--fixed_lag", action="store_true",
                      help="fixing a saving lag")
    parser.add_argument("--save_model", action="store_true",
                      help="whether to save model to disk (may be big)")

    if argv is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args(argv)
    run_m0(**vars(args))


def run_m0(**kwargs):
    options = dict((k, v) for (k, v) in DEFAULT_OPTIONS.items())
    options.update(kwargs)
    
    # Make output dir
    result_directory = options['output_dir']
    if not os.path.isdir(result_directory):
        os.makedirs(result_directory)

    # Initialize logger
    log_path = os.path.join(result_directory, LOG_BASENAME)
    logger = logging.getLogger()
    logger.setLevel(options['log_level'])
    log_file_handler = logging.FileHandler(log_path)
    log_file_handler.setFormatter(logging.Formatter(
        '%(asctime)-15s %(levelname)s %(funcName)s: %(message)s'))
    logger.addHandler(log_file_handler)

    # Add concise log handler to stdout
    log_out_handler = logging.StreamHandler(sys.stdout)
    log_out_handler.setFormatter(logging.Formatter('%(message)s'))
    log_out_handler.setLevel(logging.INFO)
    logger.addHandler(log_out_handler)

    # Write options to log, file
    options_filename = os.path.join(result_directory, OPTIONS_BASENAME)
    with open(options_filename, 'w') as options_f:
        for (k, v) in options.items():
            line = '%s: %s' % (k, v)
            logging.info(line)
            options_f.write(line + '\n')

    # Set the random seed.
    if options['random_seed'] is not None:
        m0.set_random_seed(options['random_seed'])

    if options['streaming']:
        if options['D'] is None:
            raise ValueError('D must be specified in streaming mode')
        if options['W'] is None:
            raise ValueError('W must be specified in streaming mode')
        num_docs = options['D']
        num_types = options['W']
        # TODO multiple files?
        train_file = open(options['data_path'])
        c_train = Corpus.from_stream_data(train_file, num_docs)
    else:
        train_filenames = glob(options['data_path'])
        train_filenames.sort()

        if options['D'] is None:
            num_docs = 0
            for train_filename in train_filenames:
                num_docs += len(Corpus.from_data(train_filename).docs)
        else:
            num_docs = options['D']

        if options['W'] is None:
            num_types = 0
            for train_filename in train_filenames:
                num_types = max(
                    num_types,
                    max(max(d.words) for d in
                        Corpus.from_data(train_filename).docs) + 1)
            if options['test_data_path'] is not None:
                num_types = max(
                    num_types,
                    max(max(d.words) for d in
                        Corpus.from_data(options['test_data_path']).docs) + 1)
        else:
            num_types = options['W']

        c_train = Corpus.from_data(train_filenames)

    logging.info('No. docs: %d' % num_docs)
    logging.info('No. types: %d' % num_types)

    if options['test_data_path'] is not None:
        test_data_path = options['test_data_path']
        (c_test_train, c_test_test) = Corpus.from_data(test_data_path).split_within_docs(options['test_train_frac'])

    trunc = tuple(int(t) for t in options['trunc'].split(','))

    logging.info("Creating online nhdp instance")
    model = m0.m0(trunc, num_docs, num_types,
                  options['lambda0'], options['beta'], options['alpha'],
                  options['gamma1'], options['gamma2'],
                  options['kappa'], options['iota'], options['delta'],
                  options['scale'], options['adding_noise'])

    if options['init_samples'] is not None:
        logging.info("Initializing")
        init_docs = take(c_train.docs, options['init_samples'])
        model.initialize(init_docs, options['xi'], options['omicron'])

    iteration = 0
    total_doc_count = 0

    start_time = time.time()
    logging.info("Starting online variational inference")
    while True:
        iteration += 1
        logging.info("Iteration %d" % iteration)

        # Sample the documents.
        batchsize = options['batchsize']
        if options['streaming']:
            docs = take(c_train.docs, batchsize)
        else:
            ids = random.sample(range(c_train.num_docs), batchsize)
            docs = [c_train.docs[idx] for idx in ids]

        total_doc_count += batchsize

        # Do online inference and evaluate on the fly dataset
        (score, count, doc_count) = model.process_documents(docs,
            options['var_converge'])
        logging.info('Cumulative doc count: %d' % total_doc_count)
        logging.info('Log-likelihood: %f (%f per token) (%d tokens)' % (score, score/count, count))

        # Evaluate on the test data: fixed and folds
        if total_doc_count % options['save_lag'] == 0:
            if not options['fixed_lag']:
                options['save_lag'] = options['save_lag'] * 2

            # Save the model.
            if options['save_model']:
                topics_filename = os.path.join(result_directory,
                    'doc_count-%d.topics' % total_doc_count)
                model.save_topics(topics_filename)
                model_filename = os.path.join(result_directory,
                    'doc_count-%d.model' % total_doc_count)
                with open(model_filename, 'w') as model_f:
                    cPickle.dump(model, model_f, -1)

            if options['test_data_path'] is not None:
                test_nhdp_predictive(model, c_test_train, c_test_test, batchsize, options['var_converge'], options['test_samples'])

        if options['max_iter'] is not None and iteration > options['max_iter']:
            break
        if options['max_time'] is not None and time.time() - start_time > options['max_time']:
            break

    if options['save_model']:
        logging.info("Saving the final model and topics")
        topics_filename = os.path.join(result_directory, 'final.topics')
        model.save_topics(topics_filename)
        model_filename = os.path.join(result_directory, 'final.model')
        with open(model_filename, 'w') as model_f:
            cPickle.dump(model, model_f, -1)

    if options['streaming']:
        train_file.close()

    # Makeing final predictions.
    if options['test_data_path'] is not None:
        test_nhdp_predictive(model, c_test_train, c_test_test, batchsize, options['var_converge'], options['test_samples'])


def test_nhdp(model, c, batchsize, var_converge, test_samples=None):
    total_score = 0.0
    total_count = 0

    docs_generator = (d for d in c.docs)

    if test_samples is not None:
        docs_generator = take(docs_generator, test_samples)

    doc_count = batchsize
    while doc_count == batchsize:
        batch = take(docs_generator, batchsize)

        (score, count, doc_count) = model.process_documents(
            batch, var_converge, update=False)
        total_score += score
        total_count += count

    if total_count > 0:
        logging.info('Test log-likelihood: %f (%f per token) (%d tokens)'
            % (total_score, total_score/total_count, total_count))
    else:
        logging.warn('Cannot test: no data')


def test_nhdp_predictive(model, c_train, c_test, batchsize, var_converge, test_samples=None):
    total_score = 0.0
    total_count = 0

    # need a generator or we will start over at beginning each batch
    train_docs_generator = (d for d in c_train.docs)
    test_docs_generator = (d for d in c_test.docs)

    if test_samples is not None:
        train_docs_generator = take(train_docs_generator, test_samples)
        test_docs_generator = take(test_docs_generator, test_samples)

    doc_count = batchsize
    while doc_count == batchsize:
        train_batch = take(train_docs_generator, batchsize)
        test_batch = take(test_docs_generator, batchsize)

        (score, count, doc_count) = model.process_documents(
            train_batch, var_converge, update=False, predict_docs=test_batch)
        total_score += score
        total_count += count

    if total_count > 0:
        logging.info('Test log-likelihood: %f (%f per token) (%d tokens)'
            % (total_score, total_score/total_count, total_count))
    else:
        logging.warn('Cannot test: no data')


if __name__ == '__main__':
    main()
