#!/usr/bin/env python
#$ -cwd
#$ -j y
#$ -V
#$ -N "m1"
#$ -q text.q
#$ -l num_proc=1,mem_free=2G,h_rt=4:00:00


import codecs
import logging
import time
import os
import sys
from pylowl.proj.brightside.corpus import Corpus, load_vocab
from pylowl.proj.brightside.utils import take, nested_file_paths, make_output_dir
from pylowl.proj.brightside.m1.core import *
from pylowl.proj.brightside.m1.postproc.utils import postprocess
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
        'subtree_doc_lambda_ss',
    )
]

DEFAULT_OUTPUT_PARENT_DIR = 'output/pylowl/proj/brightside/m1'

DEFAULT_OPTIONS = dict(
    log_level='INFO',
    trunc='1,7,5',
    D=None,
    U=None,
    lambda0=0.01,
    beta=1.0,
    alpha=1.0,
    gamma1=1.0,
    gamma2=1.0,
    kappa=0.6,
    iota=64.0,
    delta=1e-3,
    eff_init_samples=None,
    init_noise_weight=0.5,
    batchsize=100,
    max_iter=None,
    max_time=3600,
    var_converge=0.0001,
    random_seed=None,
    data_dir=None,
    test_data_dir=None,
    output_dir=None,
    test_samples=None,
    test_train_frac=0.9,
    save_lag=500,
    scale=1.0,
    streaming=False,
    fixed_lag=False,
    save_model=True,
    init_samples=None,
    vocab_path=None,
    concrete_section_segmentation=0,
    concrete_sentence_segmentation=0,
    concrete_tokenization_list=0,
    user_subtree_selection_interval=10,
    user_doc_reservoir_capacity=10,
)

GRID_VAR_NAME_TYPE_PAIRS = (
    ('trunc', str),
    ('alpha', float),
    ('beta', float),
    ('gamma1', float),
    ('gamma2', float),
    ('lambda0', float),
    ('kappa', float),
    ('iota', float),
    ('batchsize', int),
)


def make_arg_parser():
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
                      help="max time in seconds for training")
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
                      help="output directory (None: auto, subdir of '%s')" % DEFAULT_OUTPUT_PARENT_DIR)
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

    return parser


def main():
    parser = make_arg_parser()
    run(**vars(parser.parse_args()))


def run(**kwargs):
    options = dict((k, v) for (k, v) in DEFAULT_OPTIONS.items())
    options.update(kwargs)

    # Make output dir
    if options['output_dir'] is None:
        output_dir = make_output_dir(DEFAULT_OUTPUT_PARENT_DIR)
    else:
        output_dir = options['output_dir']
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    # Initialize logger
    log_path = os.path.join(output_dir, LOG_BASENAME)
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
    options_filename = os.path.join(output_dir, OPTIONS_BASENAME)
    with wrap_open(options_filename, 'w') as options_f:
        for (k, v) in options.items():
            line = '%s: %s' % (k, v)
            logging.info(line)
            options_f.write(line + '\n')

    # Set the random seed.
    if options['random_seed'] is not None:
        set_random_seed(options['random_seed'])

    if options['data_dir'] is None:
        raise ValueError('data_dir must be specified')
    if options['vocab_path'] is None:
        raise ValueError('vocab_path must be specified')

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
            c_test = Corpus.from_concrete(
                test_filenames, r_vocab,
                options['concrete_section_segmentation'],
                options['concrete_sentence_segmentation'],
                options['concrete_tokenization_list'],
            )
            (c_test_train, c_test_test) = c_test.split_within_docs(options['test_train_frac'])

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
            c_test = Corpus.from_concrete(
                test_filenames, r_vocab,
                options['concrete_section_segmentation'],
                options['concrete_sentence_segmentation'],
                options['concrete_tokenization_list'],
            )
            (c_test_train, c_test_test) = c_test.split_within_docs(options['test_train_frac'])

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

    batchsize = options['batchsize']

    save_and_test(m, 'initial', output_dir, options['save_model'],
                  options['test_data_dir'], c_test, c_test_train, c_test_test,
                  batchsize, options['var_converge'], options['test_samples'])

    iteration = 0
    total_num_docs = 0

    start_time = time.time()
    logging.info("Starting online variational inference")
    while True:
        iteration += 1
        logging.info("Iteration %d" % iteration)

        # Sample the documents.
        if options['streaming']:
            docs = take(c_train.docs, batchsize)
        else:
            ids = random.sample(range(c_train.num_docs), batchsize)
            docs = [c_train.docs[idx] for idx in ids]

        total_num_docs += batchsize

        # Do online inference and evaluate on the fly dataset
        (likelihood, count, num_docs, doc_likelihoods, doc_counts) = m.process_documents(docs,
            options['var_converge'])
        logging.info('Cumulative doc count: %d' % total_num_docs)
        logging.info('Score: %f (%f per token) (%d tokens)' % (likelihood, likelihood/count, count))

        if total_num_docs % options['save_lag'] == 0:
            if not options['fixed_lag']:
                options['save_lag'] = options['save_lag'] * 2

            save_and_test(m, 'num_docs-%d' % total_num_docs,
                          output_dir, options['save_model'],
                          options['test_data_dir'], c_test, c_test_train, c_test_test,
                          batchsize, options['var_converge'],
                          options['test_samples'])

        if options['max_iter'] is not None and iteration > options['max_iter']:
            break

        delta_time = time.time() - start_time
        logging.info('Elapsed time %d s' % delta_time)
        if options['max_time'] is not None and delta_time > options['max_time']:
            break

    save_and_test(m, 'final', output_dir, options['save_model'],
                  options['test_data_dir'], c_test, c_test_train, c_test_test,
                  batchsize, options['var_converge'], options['test_samples'])

    logging.info('Postprocessing output: %s' % output_dir)
    postprocess(output_dir)

    logging.info('Done.')


def save_and_test(m, basename_stem, output_dir, save_model,
                  test_data_dir, c_test, c_test_train, c_test_test, batchsize,
                  var_converge, test_samples):
    if save_model:
        logging.info('Saving model with stem %s' % basename_stem)
        output_files = make_output_files(basename_stem, output_dir)
    else:
        output_files = dict()

    if test_data_dir is not None:
        test_nhdp_rank(m, c_test, batchsize, var_converge,
                       test_samples)
        #test_nhdp_predictive(m, c_test_train, c_test_test, batchsize,
        #                     var_converge, test_samples, output_files)

    if save_model:
        m.save(output_files)
        close_output_files(output_files)


def make_output_files(basename_stem, output_dir):
    return dict(
        (s, wrap_open(os.path.join(output_dir, basename_stem + ext),
                      mode))
        for (s, ext, mode) in OUTPUT_EXTS
    )


def close_output_files(output_files):
    for (s, f) in output_files.items():
        if f is not None:
            f.close()


def test_nhdp(m, c, batchsize, var_converge, test_samples=None):
    total_likelihood = 0.0
    total_count = 0

    docs_generator = (d for d in c.docs)

    if test_samples is not None:
        docs_generator = take(docs_generator, test_samples)

    num_docs = batchsize
    while num_docs == batchsize:
        batch = take(docs_generator, batchsize)

        batch = [doc for doc in batch if doc.attrs['user'] in m.m_r_users]

        (likelihood, count, num_docs, doc_likelihoods, doc_counts) = m.process_documents(
            batch, var_converge, update=False)
        total_likelihood += likelihood
        total_count += count

    if total_count > 0:
        logging.info('Test log-likelihood: %f (%f per token) (%d tokens)'
            % (total_likelihood, total_likelihood/total_count, total_count))
    else:
        logging.warn('Cannot test: no data')


def test_nhdp_predictive(m, c_train, c_test, batchsize, var_converge, test_samples=None, output_files=None):
    total_likelihood = 0.0
    total_count = 0

    # need a generator or we will start over at beginning each batch
    train_docs_generator = (d for d in c_train.docs)
    test_docs_generator = (d for d in c_test.docs)

    if test_samples is not None:
        train_docs_generator = take(train_docs_generator, test_samples)
        test_docs_generator = take(test_docs_generator, test_samples)

    num_docs = batchsize
    while num_docs == batchsize:
        train_batch = take(train_docs_generator, batchsize)
        test_batch = take(test_docs_generator, batchsize)

        train_batch = [doc for doc in train_batch if doc.attrs['user'] in m.m_r_users]
        test_batch = [doc for doc in test_batch if doc.attrs['user'] in m.m_r_users]

        (likelihood, count, num_docs, doc_likelihoods, doc_counts) = m.process_documents(
            train_batch, var_converge, update=False, predict_docs=test_batch,
            output_files=output_files)
        total_likelihood += likelihood
        total_count += count

    if total_count > 0:
        logging.info('Test predictive log-likelihood: %f (%f per token) (%d tokens)'
            % (total_likelihood, total_likelihood/total_count, total_count))
    else:
        logging.warn('Cannot test: no data')


def test_nhdp_rank(m, c, batchsize, var_converge, test_samples=None):
    rank = 10

    total_likelihood = 0.0
    total_count = 0

    # need a generator or we will start over at beginning each batch
    docs_generator = (d for d in c.docs)

    if test_samples is not None:
        docs_generator = take(docs_generator, test_samples)

    all_doc_users = []
    all_doc_likelihoods = dict((c, []) for c in m.m_r_users)
    orig_num_docs = batchsize
    while orig_num_docs == batchsize:
        batch = list(take(docs_generator, batchsize))
        orig_num_docs = len(batch)
        batch = [doc for doc in batch if doc.attrs['user'] in m.m_r_users]

        orig_users = [doc.attrs['user'] for doc in batch]
        all_doc_users.extend(orig_users)

        for cur_user in m.m_r_users:
            for doc in batch:
                doc.attrs['user'] = cur_user

            (likelihood, count, num_docs, doc_likelihoods, doc_counts) = m.process_documents(
                batch, var_converge, update=False)
            doc_likelihoods_per_word = [
                ((doc_count > 0) and (doc_likelihood/float(doc_count)) or 0.)
                for (doc_likelihood, doc_count)
                in zip(doc_likelihoods, doc_counts)
            ]
            all_doc_likelihoods[cur_user].extend(doc_likelihoods_per_word)
            total_count += count

        for (doc, orig_user) in zip(batch, orig_users):
            doc.attrs['user'] = orig_user

    if total_count > 0:
        for cur_user in m.m_r_users:
            true_pos = 0
            false_pos = 0
            cur_user_doc_likelihoods = all_doc_likelihoods[cur_user]
            doc_idx_likelihood_pairs = sorted(
                enumerate(cur_user_doc_likelihoods),
                key=lambda p: p[1],
                reverse=True
            )
            for (doc_idx, doc_likelihood) in doc_idx_likelihood_pairs[:rank]:
                if all_doc_users[doc_idx] == cur_user:
                    true_pos += 1
                else:
                    false_pos += 1
            total_pos = true_pos + false_pos
            if total_pos == 0:
                logging.info(u'Test precision at rank %d for %s: undefined'
                             % (rank, cur_user))
            else:
                precision = true_pos / float(total_pos)
                logging.info(u'Test precision at rank %d for %s: %f'
                             % (rank, cur_user, precision))
    else:
        logging.warn('Cannot test: no data')


def wrap_open(path, mode):
    if 'b' in mode:
        return open(path, mode)
    else:
        return codecs.open(path, mode=mode, encoding='utf-8')


if __name__ == '__main__':
    main()
