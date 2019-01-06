from datetime import datetime
from sklearn import metrics
from theano import tensor as T
import argparse
import cPickle
import numpy
import os
import sys
import theano
import time
from collections import defaultdict
import subprocess
import pandas as pd
from tqdm import tqdm

import nn_layers
import sgd_trainer

import warnings
warnings.filterwarnings("ignore")  # TODO remove

### THEANO DEBUG FLAGS
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = 'high'

def paired_substract(x, y):
    return x.reshape((x.shape[0], 1, -1)) - y.reshape((1, y.shape[0], -1))

def paired_multiply(x, y):
    return x.reshape((x.shape[0], 1, -1)) * y.reshape((1, y.shape[0], -1))

def normalize(x):
    denom = (x ** 2).sum(axis=1, keepdims=True).sqrt()
    return T.switch(T.eq(denom, 0), 0, x / denom)

def L1_norm(d):
    return abs(d).sum(axis=2)

def L2_norm(d):
    return (d ** 2).sum(axis=2).sqrt()

def euclidean_distance(x, y):
    d = paired_substract(x, y)
    return (d ** 2).sum(axis=2).sqrt()

def euclidean_similarity(x, y):
    return 1.0 / (1.0 + euclidean_distance(x, y))

def cosine_similarity(x, y):
    nx = normalize(x)
    ny = normalize(y)
    nx_ny = paired_multiply(nx, ny)
    return nx_ny.sum(axis=2)


class AttentionWeightingLayer(nn_layers.Layer):
    def __init__(self, similarity):
        super(AttentionWeightingLayer, self).__init__()

        if similarity in ['euclidean']:
            self.sim_func = euclidean_similarity
        elif similarity in ['cosine']:
            self.sim_func = cosine_similarity
        else:
            raise Exception('Unknown similarity function: {}'.format(similarity))

    def output_func(self, input_):
        in_q, in_a = input_

        F_q = in_q.dimshuffle(2, 3, 0, 1).flatten(3).dimshuffle(2, 0, 1)
        F_a = in_a.dimshuffle(2, 3, 0, 1).flatten(3).dimshuffle(2, 0, 1)
        A, _ = theano.scan(fn=self.sim_func, sequences=[F_q, F_a])
        R_q, _ = theano.scan(lambda fq, a: fq * a.sum(axis=1, keepdims=True), sequences=[F_q, A])
        R_a, _ = theano.scan(lambda fa, a: fa * T.transpose(a).sum(axis=1, keepdims=True), sequences=[F_a, A])
        out_q = R_q.reshape(in_q.shape)
        out_a = R_a.reshape(in_a.shape)

        return out_q, out_a

    def __repr__(self):
        return '{}: sim_func={}'.format(self.__class__.__name__, self.sim_func.__name__)


class AttentionTransformLayer(nn_layers.Layer):
    def __init__(self, similarity, W_q=None, W_a=None, rng=None, W_q_shape=None, W_a_shape=None):
        super(AttentionTransformLayer, self).__init__()

        if similarity in ['euclidean']:
            self.sim_func = euclidean_similarity
        elif similarity in ['cosine']:
            self.sim_func = cosine_similarity
        else:
            assert self.sim_func, 'Unknown similarity function: {}'.format(similarity)

        if not W_q:
            W_q_values = numpy.asarray(
                rng.uniform(low=-numpy.sqrt(6. / (W_q_shape[0] + W_q_shape[1])),
                            high=numpy.sqrt(6. / (W_q_shape[0] + W_q_shape[1])),
                            size=W_q_shape),
                dtype=theano.config.floatX)
            W_q = theano.shared(value=W_q_values, name='W_q', borrow=True)

        if not W_a:
            W_a_values = numpy.asarray(
                rng.uniform(low=-numpy.sqrt(6. / (W_a_shape[0] + W_a_shape[1])),
                            high=numpy.sqrt(6. / (W_a_shape[0] + W_a_shape[1])),
                            size=W_a_shape),
                dtype=theano.config.floatX)
            W_a = theano.shared(value=W_a_values, name='W_a', borrow=True)

        self.W_q = W_q
        self.W_a = W_a
        self.weights = [self.W_q, self.W_a]
        self.params = [self.W_q, self.W_a]
        self.num_params = sum([numpy.prod(p.shape.eval()) for p in self.params])

    def output_func(self, input_):
        in_q, in_a = input_

        F_q = in_q.dimshuffle(2, 3, 0, 1).flatten(3).dimshuffle(2, 0, 1)
        F_a = in_a.dimshuffle(2, 3, 0, 1).flatten(3).dimshuffle(2, 0, 1)

        # A, _ = theano.scan(lambda fq, fa: 1. / (1. + euclidean_distance(fq, fa)), sequences=[F_q, F_a])
        A, _ = theano.scan(fn=self.sim_func, sequences=[F_q, F_a])
        R_q, _ = theano.scan(lambda a: T.dot(a, self.W_q), sequences=[A])
        R_a, _ = theano.scan(lambda a: T.dot(a.T, self.W_a), sequences=[A])
        out_q = T.join(1, in_q, R_q.reshape(in_q.shape))
        out_a = T.join(1, in_a, R_a.reshape(in_a.shape))

        return out_q, out_a

    def __repr__(self):
        return '{}: sim_func={} [num params: {}]'.format(self.__class__.__name__, self.sim_func.__name__, self.num_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', choices=['abcnn1', 'abcnn2'])
    parser.add_argument('--similarity', choices=['euclidean', 'cosine'])
    parser.add_argument('--no-features', action='store_true',
                        help='do not use external features')
    parser.add_argument('--l2svm', action='store_true',
                        help='use L2-SVM as the classifier')
    parser.add_argument('--dropout', choices=['gaussian', 'mc'])
    parser.add_argument('--dropout-rate', type=float,
                        help='dropout rate (default: %(default)s)')
    parser.add_argument('--nkernels', type=int,
                        help='number of kernels (default: %(default)s)')
    parser.add_argument('--early-stop', metavar='N', type=int,
                        help='stop if seeing no improvements in N epochs')
    parser.add_argument('-e', choices=['GoogleNews', 'aquaint+wiki'],
                        help='word embeddings file to use')
    parser.add_argument('mode')
    parser.set_defaults(early_stop=3, e='GoogleNews', dropout_rate=0.5, nkernels=100)
    args = parser.parse_args()

    # ZEROUT_DUMMY_WORD = False
    ZEROUT_DUMMY_WORD = True

    ## Load data
    # mode = 'TRAIN-ALL'
    mode = args.mode
    if mode not in ['TRAIN', 'TRAIN-ALL', 'WIKIQA-TRAIN'] + ['WEBAP-FOLD{}-TRAIN'.format(i) for i in (1, 2, 3, 4, 5)]:
        print "ERROR! mode '{}' is invalid".format(mode)
        sys.exit(1)

    print "Running training in the {} setting".format(mode)

    data_dir = mode

    def load_numpy_data(data_dir, prefix):
        filetypes = ['questions', 'answers', 'q_overlap_indices', 'a_overlap_indices', 'labels', 'qids', 'aids']
        filenames = ['{}.{}.npy'.format(prefix, filetype) for filetype in filetypes]
        return [numpy.load(os.path.join(data_dir, filename)) for filename in filenames]

    if mode in ['TRAIN-ALL', 'TRAIN']:
        prefix = mode.lower()
        q_train, a_train, q_overlap_train, a_overlap_train, y_train, _, _ = load_numpy_data(data_dir, prefix)
        q_dev, a_dev, q_overlap_dev, a_overlap_dev, y_dev, qids_dev, _ = load_numpy_data(data_dir, 'dev')
        q_test, a_test, q_overlap_test, a_overlap_test, y_test, qids_test, aids_test = load_numpy_data(data_dir, 'test')

        x_train = numpy.load(os.path.join(data_dir, '{}.overlap_feats.npy'.format(prefix)))
        x_dev = numpy.load(os.path.join(data_dir, 'dev.overlap_feats.npy'))
        x_test = numpy.load(os.path.join(data_dir, 'test.overlap_feats.npy'))

    elif mode in ['WIKIQA-TRAIN']:
        q_train, a_train, q_overlap_train, a_overlap_train, y_train, _, _ = load_numpy_data(data_dir, 'WikiQA-train')
        q_dev, a_dev, q_overlap_dev, a_overlap_dev, y_dev, qids_dev, _ = load_numpy_data(data_dir, 'WikiQA-dev-filtered')
        q_test, a_test, q_overlap_test, a_overlap_test, y_test, qids_test, aids_test = load_numpy_data(data_dir, 'WikiQA-test-filtered')

        x_train = numpy.load(os.path.join(data_dir, 'WikiQA-train.overlap_feats.npy'))
        x_dev = numpy.load(os.path.join(data_dir, 'WikiQA-dev-filtered.overlap_feats.npy'))
        x_test = numpy.load(os.path.join(data_dir, 'WikiQA-test-filtered.overlap_feats.npy'))

    elif mode in ['WEBAP-FOLD{}-TRAIN'.format(i) for i in (1, 2, 3, 4, 5)]:
        fn = ['WEBAP-FOLD{}-TRAIN'.format(i) for i in (1, 2, 3, 4, 5)].index(mode) + 1

        q_train, a_train, q_overlap_train, a_overlap_train, y_train, _, _ = load_numpy_data(data_dir, 'WebAP-fold{}-train'.format(fn))
        q_dev, a_dev, q_overlap_dev, a_overlap_dev, y_dev, qids_dev, _ = load_numpy_data(data_dir, 'WebAP-fold{}-dev'.format(fn))
        q_test, a_test, q_overlap_test, a_overlap_test, y_test, qids_test, aids_test = load_numpy_data(data_dir, 'WebAP-fold{}-test'.format(fn))

    # x_train = numpy.load(os.path.join(data_dir, 'train.overlap_feats.npy'))
    # x_dev = numpy.load(os.path.join(data_dir, 'dev.overlap_feats.npy'))
    # x_test = numpy.load(os.path.join(data_dir, 'test.overlap_feats.npy'))

    feats_ndim = x_train.shape[1]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(copy=True)
    print "Scaling features"
    x_train = scaler.fit_transform(x_train)
    x_dev = scaler.transform(x_dev)
    x_test = scaler.transform(x_test)

    print 'y_train', numpy.unique(y_train, return_counts=True)
    print 'y_dev', numpy.unique(y_dev, return_counts=True)
    print 'y_test', numpy.unique(y_test, return_counts=True)

    print 'q_train', q_train.shape
    print 'q_dev', q_dev.shape
    print 'q_test', q_test.shape

    print 'a_train', a_train.shape
    print 'a_dev', a_dev.shape
    print 'a_test', a_test.shape

    print 'x_train', x_train.shape
    print 'x_dev', x_dev.shape
    print 'x_test', x_test.shape

    ## Get the word embeddings from the nnet trained on SemEval
    # ndim = 40
    # nnet_outdir = 'exp/ndim=60;batch=100;max_norm=0;learning_rate=0.1;2014-12-02-15:53:14'
    # nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
    # params_fname = os.path.join(nnet_outdir, 'best_dev_params.epoch=00;batch=14640;dev_f1=83.12;test_acc=85.00.dat')
    # train_nnet, test_nnet = nn_layers.load_nnet(nnet_fname, params_fname)

    numpy_rng = numpy.random.RandomState(123)
    q_max_sent_size = q_train.shape[1]
    a_max_sent_size = a_train.shape[1]
    # print 'max', numpy.max(a_train)
    # print 'min', numpy.min(a_train)

    ndim = 5
    print "Generating random vocabulary for word overlap indicator features with dim:", ndim
    dummy_word_id = numpy.max(a_overlap_train)
    # vocab_emb_overlap = numpy_rng.uniform(-0.25, 0.25, size=(dummy_word_id+1, ndim))
    print "Gaussian"
    vocab_emb_overlap = numpy_rng.randn(dummy_word_id + 1, ndim) * 0.25
    # vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, ndim) * 0.05
    # vocab_emb_overlap = numpy_rng.uniform(-0.25, 0.25, size=(dummy_word_id+1, ndim))
    vocab_emb_overlap[-1] = 0

    # Load word2vec embeddings
    if args.e in ['GoogleNews']:
        fname = os.path.join(data_dir, 'emb_GoogleNews-vectors-negative300.bin.npy')
    elif args.e in ['aquaint+wiki']:
        fname = os.path.join(data_dir, 'emb_aquaint+wiki.txt.gz.ndim=50.bin.npy')
    else:
        print 'No such embedding file: {}'.format(args.e)
        sys.exit(1)

    print "Loading word embeddings from", fname
    vocab_emb = numpy.load(fname)
    ndim = vocab_emb.shape[1]
    dummpy_word_idx = numpy.max(a_train)
    print "Word embedding matrix size:", vocab_emb.shape

    x = T.dmatrix('x')
    x_q = T.lmatrix('q')
    x_q_overlap = T.lmatrix('q_overlap')
    x_a = T.lmatrix('a')
    x_a_overlap = T.lmatrix('a_overlap')
    y = T.ivector('y')

    #######
    n_outs = 2

    n_epochs = 25
    batch_size = 50
    learning_rate = 0.1
    max_norm = 0

    print 'batch_size', batch_size
    print 'n_epochs', n_epochs
    print 'learning_rate', learning_rate
    print 'max_norm', max_norm

    ## 1st conv layer.
    ndim = vocab_emb.shape[1] + vocab_emb_overlap.shape[1]

    ### Nonlinearity type
    # activation = nn_layers.relu_f
    activation = T.tanh

    dropout_rate = args.dropout_rate
    nkernels = args.nkernels
    q_k_max = 1
    a_k_max = 1

    # filter_widths = [3,4,5]
    q_filter_widths = [5]
    a_filter_widths = [5]

    # Lookup layers
    lookup_table_q = nn_layers.ParallelLookupTable(
        layers=[nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths) - 1),
                nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths) - 1)])
    lookup_table_q.set_input((x_q, x_q_overlap))

    lookup_table_a = nn_layers.ParallelLookupTable(
        layers=[nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(a_filter_widths) - 1),
                nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(a_filter_widths) - 1)])
    lookup_table_a.set_input((x_a, x_a_overlap))

    # NOTE: these seemingly mismatched shapes are actually correct
    if args.a in ['abcnn1']:
        attention = AttentionTransformLayer(similarity=args.similarity,
                                            rng=numpy_rng,
                                            W_q_shape=(a_max_sent_size + 2 * (max(a_filter_widths) - 1), ndim),
                                            W_a_shape=(q_max_sent_size + 2 * (max(q_filter_widths) - 1), ndim))
        num_input_channels = 2
    elif args.a in ['abcnn2']:
        attention = AttentionWeightingLayer(similarity=args.similarity)
        num_input_channels = 1
    else:
        attention = None
        num_input_channels = 1

    if attention is not None:
        attention.set_input((lookup_table_q.output, lookup_table_a.output))
        input0, input1 = attention.output
    else:
        input0, input1 = lookup_table_q.output, lookup_table_a.output

    input_shape_q = (batch_size, num_input_channels, q_max_sent_size + 2 * (max(q_filter_widths) - 1), ndim)
    input_shape_a = (batch_size, num_input_channels, a_max_sent_size + 2 * (max(a_filter_widths) - 1), ndim)
    ###### QUESTION ######

    # lookup_table_words = nn_layers.LookupTableFastStatic(
    #     W=vocab_emb, pad=max(q_filter_widths) - 1)
    # lookup_table_overlap = nn_layers.LookupTableFast(
    #     W=vocab_emb_overlap, pad=max(q_filter_widths) - 1)
    # lookup_table = nn_layers.ParallelLookupTable(
    #     layers=[lookup_table_words, lookup_table_overlap])

    # input_shape = (batch_size, num_input_channels, q_max_sent_size + 2 *
    #                (max(q_filter_widths) - 1), ndim)

    conv_layers = []
    for filter_width in q_filter_widths:
        filter_shape = (nkernels, num_input_channels, filter_width, ndim)
        conv = nn_layers.Conv2dLayer(rng=numpy_rng,
                                     filter_shape=filter_shape,
                                     input_shape=input_shape_q)
        non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0],
                                                    activation=activation)
        pooling = nn_layers.KMaxPoolLayer(k_max=q_k_max)
        conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(
            layers=[conv, non_linearity, pooling])
        conv_layers.append(conv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
    flatten_layer = nn_layers.FlattenLayer()

    nnet_q = nn_layers.FeedForwardNet(layers=[join_layer, flatten_layer])
    nnet_q.set_input(input0)
    ######

    ###### ANSWER ######
    # lookup_table_words = nn_layers.LookupTableFastStatic(
    #     W=vocab_emb, pad=max(q_filter_widths) - 1)
    # lookup_table_overlap = nn_layers.LookupTableFast(
    #     W=vocab_emb_overlap, pad=max(q_filter_widths) - 1)

    # lookup_table = nn_layers.ParallelLookupTable(
    #     layers=[lookup_table_words, lookup_table_overlap])

    # num_input_channels = len(lookup_table.layers)
    # input_shape = (batch_size, num_input_channels, a_max_sent_size + 2 *
    #                (max(a_filter_widths) - 1), ndim)
    conv_layers = []
    for filter_width in a_filter_widths:
        filter_shape = (nkernels, num_input_channels, filter_width, ndim)
        conv = nn_layers.Conv2dLayer(rng=numpy_rng,
                                     filter_shape=filter_shape,
                                     input_shape=input_shape_a)
        non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0],
                                                    activation=activation)
        pooling = nn_layers.KMaxPoolLayer(k_max=a_k_max)
        conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(
            layers=[conv, non_linearity, pooling])
        conv_layers.append(conv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
    flatten_layer = nn_layers.FlattenLayer()

    nnet_a = nn_layers.FeedForwardNet(layers=[join_layer, flatten_layer])
    nnet_a.set_input(input1)
    #######
    # print 'nnet_q.output', nnet_q.output.ndim

    q_logistic_n_in = nkernels * len(q_filter_widths) * q_k_max
    a_logistic_n_in = nkernels * len(a_filter_widths) * a_k_max

    if args.dropout:
        if args.dropout == 'gaussian':
            dropout_q = nn_layers.FastDropoutLayer(rng=numpy_rng)
            dropout_a = nn_layers.FastDropoutLayer(rng=numpy_rng)
        elif args.dropout == 'mc':
            dropout_q = nn_layers.DropoutLayer(rng=numpy_rng, p=dropout_rate)
            dropout_a = nn_layers.DropoutLayer(rng=numpy_rng, p=dropout_rate)
        dropout_q.set_input(nnet_q.output)
        dropout_a.set_input(nnet_a.output)

    # feats_nout = 10
    # x_hidden_layer = nn_layers.LinearLayer(numpy_rng, n_in=feats_ndim, n_out=feats_nout, activation=activation)
    # x_hidden_layer.set_input(x)

    # feats_nout = feats_ndim

    ### Dropout
    # classifier = nn_layers.PairwiseLogisticWithFeatsRegression(q_in=logistic_n_in,
    #                                                   a_in=logistic_n_in,
    #                                                   n_in=feats_nout,
    #                                                   n_out=n_outs)
    # # classifier.set_input((dropout_q.output, dropout_a.output, x_hidden_layer.output))
    # classifier.set_input((dropout_q.output, dropout_a.output, x))

    # # train_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, x_hidden_layer, dropout_q, dropout_a, classifier],
    # train_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, dropout_q, dropout_a, classifier],
    #                                       name="Training nnet")

    # test_classifier = nn_layers.PairwiseLogisticWithFeatsRegression(q_in=logistic_n_in,
    #                                                         a_in=logistic_n_in,
    #                                                         n_in=feats_nout,
    #                                                         n_out=n_outs,
    #                                                         W=classifier.W,
    #                                                         W_feats=classifier.W_feats,
    #                                                         b=classifier.b)
    # # test_classifier.set_input((nnet_q.output, nnet_a.output, x_hidden_layer.output))
    # test_classifier.set_input((nnet_q.output, nnet_a.output, x))
    # # test_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, x_hidden_layer, test_classifier],
    # test_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, test_classifier],
    #                                       name="Test nnet")
    #########

    # pairwise_layer = nn_layers.PairwiseMultiOnlySimWithFeatsLayer(q_in=q_logistic_n_in,
    # pairwise_layer = nn_layers.PairwiseWithFeatsLayer(q_in=q_logistic_n_in,
    #                                                   a_in=a_logistic_n_in,
    #                                                   n_in=feats_ndim)
    # pairwise_layer = nn_layers.PairwiseOnlySimWithFeatsLayer(q_in=q_logistic_n_in,

    # pairwise_layer = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in,
    #                                                 a_in=a_logistic_n_in)
    # pairwise_layer.set_input((nnet_q.output, nnet_a.output))
    if args.no_features:
        pairwise_layer = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in,
                                                        a_in=a_logistic_n_in)
        n_in = q_logistic_n_in + a_logistic_n_in + 1
        if args.dropout:
            pairwise_layer.set_input((dropout_q.output, dropout_a.output))
        else:
            pairwise_layer.set_input((nnet_q.output, nnet_a.output))
    else:
        pairwise_layer = nn_layers.PairwiseWithFeatsLayer(q_in=q_logistic_n_in,
                                                          a_in=a_logistic_n_in,
                                                          n_in=feats_ndim)
        n_in = q_logistic_n_in + a_logistic_n_in + feats_ndim + 1
        if args.dropout:
            pairwise_layer.set_input((dropout_q.output, dropout_a.output, x))
        else:
            pairwise_layer.set_input((nnet_q.output, nnet_a.output, x))

    # n_in = q_logistic_n_in + a_logistic_n_in + feats_ndim + a_logistic_n_in
    # n_in = q_logistic_n_in + a_logistic_n_in + feats_ndim + 50
    # n_in = q_logistic_n_in + a_logistic_n_in + feats_ndim + 1
    # n_in = q_logistic_n_in + a_logistic_n_in + 1
    # n_in = feats_ndim + 1
    # n_in = feats_ndim + 50

    hidden_layer = nn_layers.LinearLayer(
        numpy_rng, n_in=n_in,
        n_out=n_in, activation=activation)
    hidden_layer.set_input(pairwise_layer.output)

    if args.l2svm:
        classifier = nn_layers.L2SVM(n_in=n_in, n_out=n_outs)
    else:
        classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
    classifier.set_input(hidden_layer.output)

    all_layers = []
    if args.a:
        all_layers.append(attention)
    all_layers.extend([nnet_q, nnet_a])
    if args.dropout:
        all_layers.extend([dropout_q, dropout_a])
    all_layers.extend([pairwise_layer, hidden_layer, classifier])

    train_nnet = nn_layers.FeedForwardNet(
        layers=all_layers,
        # train_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, x_hidden_layer, classifier],
        name="Training nnet")
    test_nnet = train_nnet
    #######

    print train_nnet

    params = train_nnet.params

    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    nnet_outdir = 'exp.out/ndim={};batch={};max_norm={};learning_rate={};{}'.format(
        ndim, batch_size, max_norm, learning_rate, ts)
    if not os.path.exists(nnet_outdir):
        os.makedirs(nnet_outdir)
    nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
    print "Saving to", nnet_fname
    cPickle.dump([train_nnet, test_nnet],
                 open(nnet_fname, 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)

    total_params = sum([numpy.prod(param.shape.eval()) for param in params])
    print 'Total params number:', total_params

    cost = train_nnet.layers[-1].training_cost(y)
    # y_train_counts = numpy.unique(y_train, return_counts=True)[1].astype(numpy.float32)
    # weights_data = numpy.sum(y_train_counts) / y_train_counts
    # weights_data_norm = numpy.linalg.norm(weights_data)
    # weights_data /= weights_data_norm
    # print 'weights_data', weights_data
    # weights = theano.shared(weights_data, borrow=True)
    # cost = train_nnet.layers[-1].training_cost_weighted(y, weights=weights)

    predictions = test_nnet.layers[-1].y_pred
    predictions_prob = test_nnet.layers[-1].p_y_given_x[:, -1]

    ### L2 regularization
    # L2_word_emb = 1e-4
    # L2_conv1d = 3e-5
    # # L2_softmax = 1e-3
    # L2_softmax = 1e-4
    # print "Regularizing nnet weights"
    # for w in train_nnet.weights:
    #   L2_reg = 0.
    #   if w.name.startswith('W_emb'):
    #     L2_reg = L2_word_emb
    #   elif w.name.startswith('W_conv1d'):
    #     L2_reg = L2_conv1d
    #   elif w.name.startswith('W_softmax'):
    #     L2_reg = L2_softmax
    #   elif w.name == 'W':
    #     L2_reg = L2_softmax
    #   print w.name, L2_reg
    #   cost += T.sum(w**2) * L2_reg

    batch_x = T.dmatrix('batch_x')
    batch_x_q = T.lmatrix('batch_x_q')
    batch_x_a = T.lmatrix('batch_x_a')
    batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')
    batch_x_a_overlap = T.lmatrix('batch_x_a_overlap')
    batch_y = T.ivector('batch_y')

    # updates = sgd_trainer.get_adagrad_updates(cost, params, learning_rate=learning_rate, max_norm=max_norm, _eps=1e-6)
    updates = sgd_trainer.get_adadelta_updates(cost,
                                               params,
                                               rho=0.95,
                                               eps=1e-6,
                                               max_norm=max_norm,
                                               word_vec_name='W_emb')

    inputs_pred = [batch_x_q,
                   batch_x_a,
                   batch_x_q_overlap,
                   batch_x_a_overlap,
                   batch_x,
                   ]

    givens_pred = {x_q: batch_x_q,
                   x_a: batch_x_a,
                   x_q_overlap: batch_x_q_overlap,
                   x_a_overlap: batch_x_a_overlap,
                   x: batch_x
                   }

    inputs_train = [batch_x_q,
                    batch_x_a,
                    batch_x_q_overlap,
                    batch_x_a_overlap,
                    batch_x,
                    batch_y, ]

    givens_train = {x_q: batch_x_q,
                    x_a: batch_x_a,
                    x_q_overlap: batch_x_q_overlap,
                    x_a_overlap: batch_x_a_overlap,
                    x: batch_x,
                    y: batch_y}

    train_fn = theano.function(inputs=inputs_train,
                               outputs=cost,
                               updates=updates,
                               givens=givens_train,
                               on_unused_input='warn')

    pred_fn = theano.function(inputs=inputs_pred,
                              outputs=predictions,
                              givens=givens_pred,
                              on_unused_input='warn')

    pred_prob_fn = theano.function(inputs=inputs_pred,
                                   outputs=predictions_prob,
                                   givens=givens_pred,
                                   on_unused_input='warn')

    def predict_batch(batch_iterator):
        preds = numpy.hstack([pred_fn(batch_x_q, batch_x_a, batch_x_q_overlap,
                                      batch_x_a_overlap, batch_x)
                              for batch_x_q, batch_x_a, batch_x_q_overlap,
                              batch_x_a_overlap, batch_x, _ in batch_iterator])
        return preds[:batch_iterator.n_samples]

    def predict_prob_batch(batch_iterator):
        preds = numpy.hstack([pred_prob_fn(
            batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, batch_x)
                              for batch_x_q, batch_x_a, batch_x_q_overlap,
                              batch_x_a_overlap, batch_x, _ in batch_iterator])
        return preds[:batch_iterator.n_samples]

    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng,
        [q_train, a_train, q_overlap_train, a_overlap_train, x_train, y_train],
        batch_size=batch_size,
        randomize=True)
    dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng, [q_dev, a_dev, q_overlap_dev, a_overlap_dev, x_dev, y_dev],
        batch_size=batch_size,
        randomize=False)
    test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, x_test, y_test],
        batch_size=batch_size,
        randomize=False)

    labels = sorted(numpy.unique(y_test))
    print 'labels', labels

    def map_score(qids, labels, preds):
        qid2cand = defaultdict(list)
        for qid, label, pred in zip(qids, labels, preds):
            qid2cand[qid].append((pred, label))

        average_precs = []
        for qid, candidates in qid2cand.iteritems():
            average_prec = 0
            running_correct_count = 0
            for i, (score, label) in enumerate(
                    sorted(candidates, reverse=True),
                    1):
                if label > 0:
                    running_correct_count += 1
                    average_prec += float(running_correct_count) / i
            average_precs.append(average_prec / (running_correct_count + 1e-6))
        map_score = sum(average_precs) / len(average_precs)
        return map_score

    print "Zero out dummy word:", ZEROUT_DUMMY_WORD
    if ZEROUT_DUMMY_WORD:
        W_emb_list = [w for w in params if w.name == 'W_emb']
        zerout_dummy_word = theano.function(
            [],
            updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

    # weights_dev = numpy.zeros(len(y_dev))
    # weights_dev[y_dev == 0] = weights_data[0]
    # weights_dev[y_dev == 1] = weights_data[1]
    # print weights_dev

    best_dev_acc = -numpy.inf
    epoch = 0
    timer_train = time.time()
    no_best_dev_update = 0
    num_train_batches = len(train_set_iterator)
    while epoch < n_epochs:
        timer = time.time()
        for i, (x_q, x_a, x_q_overlap, x_a_overlap, x, y) in enumerate(
                tqdm(train_set_iterator), 1):
            train_fn(x_q, x_a, x_q_overlap, x_a_overlap, x, y)

            # Make sure the null word in the word embeddings always remains zero
            if ZEROUT_DUMMY_WORD:
                zerout_dummy_word()

            if i % 10 == 0 or i == num_train_batches:
                y_pred_dev = predict_prob_batch(dev_set_iterator)
                # # dev_acc = map_score(qids_dev, y_dev, predict_prob_batch(dev_set_iterator)) * 100
                dev_acc = metrics.roc_auc_score(y_dev, y_pred_dev) * 100
                if dev_acc > best_dev_acc:
                    y_pred = predict_prob_batch(test_set_iterator)
                    test_acc = map_score(qids_test, y_test, y_pred) * 100

                    print(
                        'epoch: {} batch: {} dev auc: {:.4f}; test map: {:.4f}; best_dev_acc: {:.4f}'.format(
                            epoch, i, dev_acc, test_acc, best_dev_acc))
                    best_dev_acc = dev_acc
                    best_params = [numpy.copy(p.get_value(borrow=True))
                                   for p in params]
                    no_best_dev_update = 0

        if no_best_dev_update >= args.early_stop:
            print "Quitting after of no update of the best score on dev set", no_best_dev_update
            break

        print('epoch {} took {:.4f} seconds'.format(epoch,
                                                    time.time() - timer))
        epoch += 1
        no_best_dev_update += 1

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    for i, param in enumerate(best_params):
        params[i].set_value(param, borrow=True)

    y_pred_test = predict_prob_batch(test_set_iterator)
    test_acc = map_score(qids_test, y_test, y_pred_test) * 100
    fname = os.path.join(
        nnet_outdir,
        'best_dev_params.epoch={:02d};batch={:05d};dev_acc={:.2f}.dat'.format(
            epoch, i, best_dev_acc))
    numpy.savetxt(
        os.path.join(
            nnet_outdir,
            'test.epoch={:02d};batch={:05d};dev_acc={:.2f}.predictions.npy'.format(
                epoch, i, best_dev_acc)), y_pred)
    cPickle.dump(best_params,
                 open(fname, 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)

    print "Running trec_eval script..."
    N = len(y_pred_test)

    df_submission = pd.DataFrame(
        index=numpy.arange(N),
        columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
    df_submission['qid'] = qids_test
    df_submission['iter'] = 0
    df_submission['docno'] = aids_test
    df_submission['rank'] = 0
    df_submission['sim'] = y_pred_test
    df_submission['run_id'] = 'nnet'
    df_submission.to_csv(
        os.path.join(nnet_outdir, 'submission.txt'),
        header=False,
        index=False,
        sep=' ')

    df_gold = pd.DataFrame(index=numpy.arange(N),
                           columns=['qid', 'iter', 'docno', 'rel'])
    df_gold['qid'] = qids_test
    df_gold['iter'] = 0
    df_gold['docno'] = aids_test
    df_gold['rel'] = y_test
    df_gold.to_csv(
        os.path.join(nnet_outdir, 'gold.txt'),
        header=False,
        index=False,
        sep=' ')

    subprocess.call("/bin/sh run_eval.sh '{}'".format(nnet_outdir), shell=True)
    print 'results saved to directory {}'.format(nnet_outdir)


if __name__ == '__main__':
    main()
