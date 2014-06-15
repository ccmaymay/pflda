import logging
import time
import os
import sys
from data.corpus import Corpus
from data.data import take
import m0
import cPickle
import random
from optparse import OptionParser
from glob import glob


LOG_BASENAME = 'log'
OPTIONS_BASENAME = 'options.dat'


def run_m0():
    parser = OptionParser()
    parser.set_defaults(log_level='INFO',
                        trunc='1,20,10,5', D=0, W=0,
                        lambda0=0.01, beta=1.0, alpha=1.0,
                        gamma1=1/3.0, gamma2=2/3.0,
                        kappa=0.5, iota=1.0, delta=1e-3, omicron=None, xi=0.5,
                        batchsize=100,
                        max_iter=0, max_time=0,
                        var_converge=0.0001, random_seed=None,
                        data_path=None, test_data_path=None,
                        directory='output', test_samples=None,
                        test_train_frac=0.9,
                        save_lag=500, pass_ratio=0.5,
                        scale=1.0, adding_noise=False,
                        streaming=False, fixed_lag=False, save_model=False,
                        init_samples=None)

    parser.add_option("--log_level", type="string", dest="log_level",
                      help="log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) [INFO]")
    parser.add_option("--trunc", type="string", dest="trunc",
                      help="comma-separated list of truncations (per level) [20,10,5]")
    parser.add_option("--D", type="int", dest="D",
                      help="number of documents [auto]")
    parser.add_option("--W", type="int", dest="W",
                      help="size of vocabulary [auto]")
    parser.add_option("--lambda0", type="float", dest="lambda0",
                      help="the topic Dirichlet [0.01]")
    parser.add_option("--beta", type="float", dest="beta",
                      help="beta value [1.0]")
    parser.add_option("--alpha", type="float", dest="alpha",
                      help="alpha value [1.0]")
    parser.add_option("--gamma1", type="float", dest="gamma1",
                      help="gamma1 value [0.3333]")
    parser.add_option("--gamma2", type="float", dest="gamma2",
                      help="gamma2 value [0.6667]")
    parser.add_option("--kappa", type="float", dest="kappa",
                      help="learning rate [0.5]")
    parser.add_option("--iota", type="float", dest="iota",
                      help="slow down [1.0]")
    parser.add_option("--delta", type="float", dest="delta",
                      help="greedy subtree selection stopping crit [1e-3]")
    parser.add_option("--omicron", type="float", dest="omicron",
                      help="effective no. documents in initialization [actual no. documents]")
    parser.add_option("--xi", type="float", dest="xi",
                      help="fraction of topic mass derived from data in initialization [0.5]")
    parser.add_option("--batchsize", type="int", dest="batchsize",
                      help="batch size [100]")
    parser.add_option("--max_iter", type="int", dest="max_iter",
                      help="max iterations for training [no max]")
    parser.add_option("--max_time", type="int", dest="max_time",
                      help="max time in seconds for training [no max]")
    parser.add_option("--var_converge", type="float", dest="var_converge",
                      help="relative change on doc lower bound [0.0001]")
    parser.add_option("--random_seed", type="int", dest="random_seed",
                      help="the random seed [auto]")
    parser.add_option("--data_path", type="string", dest="data_path",
                      help="training data path or pattern [None]")
    parser.add_option("--test_data_path", type="string", dest="test_data_path",
                      help="testing data path [None]")
    parser.add_option("--test_train_frac", type="float", dest="test_train_frac",
                      help="fraction of testing docs on which to infer local distributions [0.9]")
    parser.add_option("--directory", type="string", dest="directory",
                      help="output directory [output]")
    parser.add_option("--save_lag", type="int", dest="save_lag",
                      help="the minimal saving lag, increasing as save_lag * 2^i, with max i as 10; default 500.")
    parser.add_option("--pass_ratio", type="float", dest="pass_ratio",
                      help="The pass ratio for each split of training data [0.5]")
    parser.add_option("--init_samples", type="int", dest="init_samples",
                      help="number of initialization documents (nested k-means init) [0]")
    parser.add_option("--test_samples", type="int", dest="test_samples",
                      help="number of test documents [auto]")
    parser.add_option("--scale", type="float", dest="scale",
                      help="scaling parameter for learning rate [1.0]")
    parser.add_option("--adding_noise", action="store_true", dest="adding_noise",
                      help="add noise to the first couple of iterations")
    parser.add_option("--streaming", action="store_true", dest="streaming",
                      help="process data in streaming fashion (D and W must be specified)")
    parser.add_option("--fixed_lag", action="store_true", dest="fixed_lag",
                      help="fixing a saving lag")
    parser.add_option("--save_model", action="store_true", dest="save_model",
                      help="whether to save model to disk (warning: big)")

    (options, args) = parser.parse_args()

    # Make output dir
    result_directory = options.directory
    if not os.path.isdir(result_directory):
        os.makedirs(result_directory)

    # Initialize logger
    log_path = os.path.join(result_directory, LOG_BASENAME)
    logger = logging.getLogger()
    logger.setLevel(options.log_level)
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
        for option in parser.option_list:
            if option.dest is not None:
                value = getattr(options, option.dest)
                line = '%s: %s' % (option.dest, value)
                logging.info(line)
                options_f.write(line + '\n')

    # Set the random seed.
    if options.random_seed is not None:
        m0.set_random_seed(options.random_seed)

    if options.streaming:
        if options.D <= 0:
            raise ValueError('D must be specified in streaming mode')
        if options.W <= 0:
            raise ValueError('W must be specified in streaming mode')
        num_docs = options.D
        num_types = options.W
        train_file = open(options.data_path)
        c_train = Corpus.from_stream_data(train_file, num_docs)
    else:
        train_filenames = glob(options.data_path)
        train_filenames.sort()
        num_train_splits = len(train_filenames)
        if options.D <= 0:
            num_docs = 0
            for train_filename in train_filenames:
                num_docs += len(Corpus.from_data(train_filename).docs)
        else:
            num_docs = options.D

        if options.W <= 0:
            num_types = 0
            for train_filename in train_filenames:
                num_types = max(
                    num_types,
                    max(max(d.words) for d in
                        Corpus.from_data(train_filename).docs) + 1)
            if options.test_data_path is not None:
                num_types = max(
                    num_types,
                    max(max(d.words) for d in
                        Corpus.from_data(options.test_data_path).docs) + 1)
        else:
            num_types = options.W

        # This is used to determine when we reload some another split.
        num_docs_per_split = num_docs / num_train_splits
        cur_chosen_split = 0
        cur_train_filename = train_filenames[cur_chosen_split]
        c_train = Corpus.from_data(cur_train_filename)

    logging.info('No. docs: %d' % num_docs)
    logging.info('No. types: %d' % num_types)

    if options.test_data_path is not None:
        test_data_path = options.test_data_path
        (c_test_train, c_test_test) = Corpus.from_data(test_data_path).split_within_docs(options.test_train_frac)

    trunc = tuple(int(t) for t in options.trunc.split(','))

    logging.info("Creating online nhdp instance")
    model = m0.m0(trunc, num_docs, num_types,
                  options.lambda0, options.beta, options.alpha,
                  options.gamma1, options.gamma2,
                  options.kappa, options.iota, options.delta,
                  options.scale, options.adding_noise)

    if options.init_samples > 0:
        init_docs = take(c_train.docs, options.init_samples)
        model.initialize(init_docs, options.xi, options.omicron)

    iteration = 0
    total_doc_count = 0
    split_doc_count = 0

    start_time = time.time()
    logging.info("Starting online variational inference")
    while True:
        iteration += 1
        logging.info("Iteration %d" % iteration)

        # Sample the documents.
        batchsize = options.batchsize
        if options.streaming:
            docs = take(c_train.docs, batchsize)
        else:
            ids = random.sample(range(c_train.num_docs), batchsize)
            docs = [c_train.docs[idx] for idx in ids]

        total_doc_count += batchsize
        split_doc_count += batchsize

        # Do online inference and evaluate on the fly dataset
        (score, count, doc_count) = model.process_documents(docs,
            options.var_converge)
        logging.info('Cumulative doc count: %d' % total_doc_count)
        logging.info('Log-likelihood: %f (%f per token) (%d tokens)' % (score, score/count, count))

        # Evaluate on the test data: fixed and folds
        if total_doc_count % options.save_lag == 0:
            if not options.fixed_lag:
                options.save_lag = options.save_lag * 2

            # Save the model.
            if options.save_model:
                topics_filename = os.path.join(result_directory,
                    'doc_count-%d.topics' % total_doc_count)
                model.save_topics(topics_filename)
                model_filename = os.path.join(result_directory,
                    'doc_count-%d.model' % total_doc_count)
                with open(model_filename, 'w') as model_f:
                    cPickle.dump(model, model_f, -1)

            if options.test_data_path is not None:
                test_nhdp_predictive(model, c_test_train, c_test_test, batchsize, options.var_converge, options.test_samples)

        # read another split.
        if not options.streaming:
            if split_doc_count > num_docs_per_split * options.pass_ratio and num_train_splits > 1:
                logging.info("Loading a new split from the training data")
                split_doc_count = 0
                # cur_chosen_split = int(random.random() * num_train_splits)
                cur_chosen_split = (cur_chosen_split + 1) % num_train_splits
                cur_train_filename = train_filenames[cur_chosen_split]
                c_train = Corpus.from_data(cur_train_filename)

        if options.max_iter > 0 and iteration > options.max_iter:
            break
        if options.max_time > 0 and time.time() - start_time > options.max_time:
            break

    if options.save_model:
        logging.info("Saving the final model and topics")
        topics_filename = os.path.join(result_directory, 'final.topics')
        model.save_topics(topics_filename)
        model_filename = os.path.join(result_directory, 'final.model')
        with open(model_filename, 'w') as model_f:
            cPickle.dump(model, model_f, -1)

    if options.streaming:
        train_file.close()

    # Makeing final predictions.
    if options.test_data_path is not None:
        test_nhdp_predictive(model, c_test_train, c_test_test, batchsize, options.var_converge, options.test_samples)


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

    logging.info('Test log-likelihood: %f (%f per token) (%d tokens)'
        % (total_score, total_score/total_count, total_count))


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

    logging.info('Test log-likelihood: %f (%f per token) (%d tokens)'
        % (total_score, total_score/total_count, total_count))


if __name__ == '__main__':
    run_m0()
