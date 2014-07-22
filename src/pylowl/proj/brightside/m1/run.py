import codecs
import logging
import time
import os
import sys
from pylowl.proj.brightside.corpus import Corpus, load_vocab
from pylowl.proj.brightside.utils import take, nested_file_paths
from pylowl.proj.brightside.m1.core import *
import random


LOG_BASENAME = 'log'

OPTIONS_BASENAME = 'options'

OUTPUT_EXTS = [
    (s, '.' + s, 'wb' if s == 'pickle' else 'w')
    for s in (
        'pickle',
        'lambda_ss',
        'Elogpi',
        'logEpi',
        'Elogtheta',
        'logEtheta',
        'subtree',
        'subtree_Elogpi',
        'subtree_logEpi',
        'subtree_lambda_ss',
    )
]

DEFAULT_OPTIONS = dict(
    log_level='INFO',
    trunc='1,20,10,5',
    D=None,
    U=None,
    lambda0=0.01,
    beta=1.0,
    alpha=1.0,
    gamma1=1.0,
    gamma2=1.0,
    kappa=0.5,
    iota=1.0,
    delta=1e-3,
    eff_init_samples=None,
    init_noise_weight=0.5,
    batchsize=100,
    max_iter=None,
    max_time=None,
    var_converge=0.0001,
    random_seed=None,
    data_dir=None,
    test_data_dir=None,
    output_dir='output',
    test_samples=None,
    test_train_frac=0.9,
    save_lag=500,
    pass_ratio=0.5,
    scale=1.0,
    streaming=False,
    fixed_lag=False,
    save_model=False,
    init_samples=None,
    vocab_path=None,
    concrete_section_segmentation=0,
    concrete_sentence_segmentation=0,
    concrete_tokenization_list=0,
    user_subtree_selection_interval=10,
    user_doc_reservoir_capacity=10,
)


def main(argv=None):
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**DEFAULT_OPTIONS)

    parser.add_argument("--log_level", type=str,
                      choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                      help="log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--trunc", type=str,
                      help="comma-separated list of truncations (per level)")
    parser.add_argument("--D", type=int,
                      help="number of documents (None: auto)")
    parser.add_argument("--U", type=int,
                      help="number of users")
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
    parser.add_argument("--eff_init_samples", type=float,
                      help="effective no. documents in initialization (None: actual)")
    parser.add_argument("--init_noise_weight", type=float,
                      help="fraction of topic mass derived from Dirichlet noise in initialization")
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
    parser.add_argument("--data_dir", type=str,
                      help="training data dir path")
    parser.add_argument("--test_data_dir", type=str,
                      help="testing data dir path")
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
    parser.add_argument("--vocab_path", type=str,
                      help="path to vocab for concrete data")
    parser.add_argument("--concrete_section_segmentation", type=int,
                      help="concrete section segmentation index")
    parser.add_argument("--concrete_sentence_segmentation", type=int,
                      help="concrete sentence segmentation index")
    parser.add_argument("--concrete_tokenization_list", type=int,
                      help="concrete tokenization list index")
    parser.add_argument("--streaming", action="store_true",
                      help="process data in streaming fashion (D must be specified)")
    parser.add_argument("--fixed_lag", action="store_true",
                      help="fixing a saving lag")
    parser.add_argument("--save_model", action="store_true",
                      help="whether to save model to disk (may be big)")
    parser.add_argument("--user_subtree_selection_interval", type=int,
                      help="interval (in docs, per user) between subtree (re)selection")
    parser.add_argument("--user_doc_reservoir_capacity", type=int,
                      help="capacity of each per-user document reservoir")

    if argv is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args(argv)
    run(**vars(args))


def run(**kwargs):
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
    with wrap_open(options_filename, 'w') as options_f:
        for (k, v) in options.items():
            line = '%s: %s' % (k, v)
            logging.info(line)
            options_f.write(line + '\n')

    # Set the random seed.
    if options['random_seed'] is not None:
        set_random_seed(options['random_seed'])

    if options['streaming']:
        if options['D'] is None:
            raise ValueError('D must be specified in streaming mode')
        num_docs = options['D']

        if options['U'] is None:
            raise ValueError('U must be specified in streaming mode')
        num_users = options['U']

        # TODO multiple files?

        train_filenames = nested_file_paths(options['data_dir'])
        train_filenames.sort()

        vocab = load_vocab(options['vocab_path'])
        num_types = len(vocab)
        r_vocab = dict((v, k) for (k, v) in vocab.items())

        c_train = Corpus.from_concrete_stream(
            train_filenames, r_vocab, num_docs,
            options['concrete_section_segmentation'],
            options['concrete_sentence_segmentation'],
            options['concrete_tokenization_list'],
        )
        if options['test_data_dir'] is not None:
            test_filenames = nested_file_paths(options['test_data_dir'])
            test_filenames.sort()
            (c_test_train, c_test_test) = Corpus.from_concrete(
                test_filenames, r_vocab,
                options['concrete_section_segmentation'],
                options['concrete_sentence_segmentation'],
                options['concrete_tokenization_list'],
            ).split_within_docs(options['test_train_frac'])

    else:
        train_filenames = nested_file_paths(options['data_dir'])
        train_filenames.sort()

        if options['D'] is None:
            num_docs = len(train_filenames)
        else:
            num_docs = options['D']

        vocab = load_vocab(options['vocab_path'])
        r_vocab = dict((v, k) for (k, v) in vocab.items())
        num_types = len(vocab)

        c_train = Corpus.from_concrete(
            train_filenames, r_vocab,
            options['concrete_section_segmentation'],
            options['concrete_sentence_segmentation'],
            options['concrete_tokenization_list'],
        )

        if options['U'] is None:
            users = set()
            for doc in c_train.docs:
                users.add(doc.attrs['user'])
            num_users = len(users)
        else:
            num_users = options['U']

        if options['test_data_dir'] is not None:
            test_filenames = nested_file_paths(options['test_data_dir'])
            test_filenames.sort()
            (c_test_train, c_test_test) = Corpus.from_concrete(
                test_filenames, r_vocab,
                options['concrete_section_segmentation'],
                options['concrete_sentence_segmentation'],
                options['concrete_tokenization_list'],
            ).split_within_docs(options['test_train_frac'])

    logging.info('No. docs: %d' % num_docs)
    logging.info('No. types: %d' % num_types)

    trunc = tuple(int(t) for t in options['trunc'].split(','))

    logging.info("Creating online nhdp instance")
    m = model(trunc, D=num_docs, W=num_types, U=num_users,
              lambda0=options['lambda0'],
              beta=options['beta'],
              alpha=options['alpha'],
              gamma1=options['gamma1'],
              gamma2=options['gamma2'],
              kappa=options['kappa'],
              iota=options['iota'],
              delta=options['delta'],
              scale=options['scale'],
              user_subtree_selection_interval=options['user_subtree_selection_interval'],
              user_doc_reservoir_capacity=options['user_doc_reservoir_capacity'])

    if options['init_samples'] is not None:
        logging.info("Initializing")
        init_docs = take(c_train.docs, options['init_samples'])
        m.initialize(init_docs, options['init_noise_weight'], options['eff_init_samples'])

    save(m, 'initial', result_directory, options['save_model'])

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
        (score, count, doc_count) = m.process_documents(docs,
            options['var_converge'])
        logging.info('Cumulative doc count: %d' % total_doc_count)
        logging.info('Log-likelihood: %f (%f per token) (%d tokens)' % (score, score/count, count))

        # Evaluate on the test data: fixed and folds
        if total_doc_count % options['save_lag'] == 0:
            if not options['fixed_lag']:
                options['save_lag'] = options['save_lag'] * 2

            # Save the model.
            save(m, 'doc_count-%d' % total_doc_count, result_directory,
                        options['save_model'])

            if options['test_data_dir'] is not None:
                test_nhdp_predictive(m, c_test_train, c_test_test, batchsize, options['var_converge'], options['test_samples'])

        if options['max_iter'] is not None and iteration > options['max_iter']:
            break

        delta_time = time.time() - start_time
        logging.info('Elapsed time %d s' % delta_time)
        if options['max_time'] is not None and delta_time > options['max_time']:
            break

    # Save the model.
    save(m, 'final', result_directory, options['save_model'])

    # Making final predictions.
    if options['test_data_dir'] is not None:
        test_nhdp_predictive(m, c_test_train, c_test_test, batchsize, options['var_converge'], options['test_samples'])


def save(m, basename_stem, result_directory, save_model):
    if save_model:
        logging.info('Saving model with stem %s' % basename_stem)
    output_files = make_output_files(basename_stem, result_directory,
                                     save_model)
    m.save(output_files)
    close_output_files(output_files)


def make_output_files(basename_stem, result_directory, save_model):
    if save_model:
        return dict(
            (s, wrap_open(os.path.join(result_directory, basename_stem + ext), mode))
            for (s, ext, mode) in OUTPUT_EXTS
        )
    else:
        return dict()


def close_output_files(output_files):
    for (s, f) in output_files.items():
        if f is not None:
            f.close()


def test_nhdp(m, c, batchsize, var_converge, test_samples=None):
    total_score = 0.0
    total_count = 0

    docs_generator = (d for d in c.docs)

    if test_samples is not None:
        docs_generator = take(docs_generator, test_samples)

    doc_count = batchsize
    while doc_count == batchsize:
        batch = take(docs_generator, batchsize)

        (score, count, doc_count) = m.process_documents(
            batch, var_converge, update=False)
        total_score += score
        total_count += count

    if total_count > 0:
        logging.info('Test log-likelihood: %f (%f per token) (%d tokens)'
            % (total_score, total_score/total_count, total_count))
    else:
        logging.warn('Cannot test: no data')


def test_nhdp_predictive(m, c_train, c_test, batchsize, var_converge, test_samples=None):
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

        (score, count, doc_count) = m.process_documents(
            train_batch, var_converge, update=False, predict_docs=test_batch)
        total_score += score
        total_count += count

    if total_count > 0:
        logging.info('Test log-likelihood: %f (%f per token) (%d tokens)'
            % (total_score, total_score/total_count, total_count))
    else:
        logging.warn('Cannot test: no data')


def wrap_open(path, mode):
    if 'b' in mode:
        return open(path, mode)
    else:
        return codecs.open(path, mode=mode, encoding='utf-8')


if __name__ == '__main__':
    main()
