"""
Feature extraction pipeline
"""
import argparse
import gensim
import krovetzstemmer
import logging
import numpy as np
import operator
import os
import porterstemmer

from smart_open import smart_open
from tqdm import tqdm

import features
import trecqa
import wikiqa

from context import SentenceSelection
from context import QREL, DOCNO, TEXT
from context import QID_QTERMS, QID_TERMS, QID_QTEXT, QID_TEXT, QID_QSTEMS, QID_STEMS


def make_feature(func, *argfuncs):
    """Return a customized feature function that adapts to different input representations.

    Args:
        func: feature function (callable)
        argfuncs: argument adaptor functions (callable, take `ctx` as input)
    """
    assert callable(func)
    for argfunc in argfuncs:
        assert callable(argfunc)

    def _feature(ctx):
        return func(*[argfunc(ctx) for argfunc in argfuncs])
    return _feature


class Pipeline():
    """Pipeline object"""

    def __init__(self, model_path='.'):
        self.features = {}
        self.queue = []
        self.model_path = model_path

    def filepath(self, filename, prefix=''):
        return os.path.join(self.model_path, '{}{}'.format(prefix, filename))

    def install_feature(self, name, *args):
        """Put a feature into the pipeline."""
        parts = []
        for arg in args:
            assert callable(arg)
            part_name = arg.__name__ if hasattr(arg, '__name__') else arg.__class__.__name__
            parts.append(part_name)

        logging.info('Install feature %s: %s' % (name, ', '.join(parts)))
        self.features[name] = make_feature(*args)
        self.queue.append(name)

    def uninstall_feature(self, name):
        """Remove a feature from the pipeline."""
        logging.info('Uninstall feature %s' % name)
        self.features.pop(name)
        self.queue.remove(name)

    def set_qrels_interface(self, arg):
        """Specify the argfunc to retrieve qrels info."""
        logging.info('Set qrels interface %s' % arg.__name__)
        self.qrels_interface = arg

    def run(self, ctx, output=None, qrels=None, overwrite=False, prefix=''):
        """Execute the pipeline."""
        self.compute_qrels(ctx, prefix)
        for name in self.queue:
            self.compute(ctx, name, overwrite, prefix)
        if output:
            self.compile_svmlight_output(output, prefix)

    def compute_qrels(self, ctx, prefix=''):
        """Compute attributes "qid", "docno", and "rel" and save them to disk."""
        qrels = list(tqdm(self.qrels_interface(ctx), desc='Compute qrels', unit=''))
        self.total = len(qrels)
        np.save(self.filepath('qid.npy', prefix), [qrel[0] for qrel in qrels])
        np.save(self.filepath('docno.npy', prefix), [qrel[1] for qrel in qrels])
        np.save(self.filepath('rel.npy', prefix), [qrel[2] for qrel in qrels])

    def compute(self, ctx, name, overwrite=False, prefix=''):
        """Compute the feature and save it to disk."""
        filepath = self.filepath('feature_{}.npy'.format(name), prefix)
        if os.path.exists(filepath) and not overwrite:
            logging.info('Compute %s: Skipped' % name)
            return
        values = np.array(list(tqdm(self.features[name](ctx),
                                    total=self.total,
                                    desc='Compute {}'.format(name),
                                    unit='')))
        assert len(values) == self.total
        np.save(filepath, values)

    def compile_svmlight_output(self, output_filename, prefix=''):
        """Compile the SVMLight format output."""
        qids = np.load(self.filepath('qid.npy', prefix))
        docnos = np.load(self.filepath('docno.npy', prefix))
        rels = np.load(self.filepath('rel.npy', prefix))

        if (rels < 0).any():
            logging.info('Negative rels are replace with 0s')
            rels[rels < 0] = 0

        vectors = []
        names = []
        for fid, name in enumerate(self.queue, 1):
            filepath = self.filepath('feature_{}.npy'.format(name), prefix)
            vector = np.load(filepath)
            if issubclass(vector.dtype.type, np.float):
                vector = vector.round(decimals=6)
            vectors.append((fid, vector))
            names.append((fid, name))

        if output_filename:
            # use a dict to keep track of qid numbers
            qid_to_number = {}

            # no need to prefix this file
            with open(self.filepath(output_filename), 'w') as out:
                for fid, name in names:
                    out.write('# %s: %s\n' % (fid, name))
                for i in tqdm(range(qids.size), desc='Write output to {}'.format(out.name), unit=''):
                    qid, docno, rel = map(operator.itemgetter(i), [qids, docnos, rels])
                    data = ' '.join(['{}:{}'.format(fid, vec[i]) for fid, vec in vectors])

                    if qid not in qid_to_number:
                        qid_to_number[qid] = len(qid_to_number) + 1

                    out.write('{rel} qid:{qid} {data} # {docno}\n'
                              .format(rel=rel, qid=qid_to_number[qid], data=data, docno=docno))


def get_stemmer(name):
    if name == 'porter':
        return porterstemmer.Stemmer()
    elif name == 'krovetz':
        return krovetzstemmer.Stemmer()
    else:
        return None


def make_stoplist(iterable):
    stoplist = set()
    if iterable:
        stoplist.update(line.strip() for line in iterable)

    def _stoplist(term):
        return term in stoplist
    return _stoplist


def run_wikiqa(ctx, pipeline, filenames, **kwargs):
    train, dev, test = filenames

    for tag, fname in zip(['train', 'dev', 'test'], filenames):
        ctx.load_data(wikiqa.get_topics(smart_open(fname, 'r')),
                      wikiqa.get_sentences_in_queries(smart_open(fname, 'r')))
        pipeline.run(ctx,
                     output='output.%s' % tag,
                     prefix='%s_' % tag,
                     **kwargs)


def run_trecqa(ctx, pipeline, filenames, **kwargs):
    # pipeline.uninstall_feature('Location')

    train_all, train, dev, test = filenames

    for tag, fname in zip(['train_all', 'train', 'dev', 'test'], filenames):
        ctx.load_data(trecqa.get_topics(smart_open(fname, 'r')),
                      trecqa.get_sentences_in_queries(smart_open(fname, 'r')))
        pipeline.run(ctx,
                     output='output.%s' % tag,
                     prefix='%s_' % tag,
                     **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='%(prog)s -m <model_path> [option..] <file>..',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=100),
    )
    parser.add_argument('-m', dest='model_path', metavar='PATH',
                        help='model path')
    parser.add_argument('--overwrite', action='store_true',
                        help='force overwrite existing feature output')
    parser.add_argument('--stemmer', metavar='NAME',
                        help='porter or krovetz (default: %(default)s)')
    parser.add_argument('--stoplist', metavar='FILE',
                        help='stoplist file (default: %(default)s)')
    parser.add_argument('--synonyms', metavar='FILE',
                        help='synonyms file (default: %(default)s)')
    parser.add_argument('--freqstats', metavar='FILE',
                        help='LM: freqstats file (default: %(default)s)')
    parser.add_argument('--lm-mu', metavar='REAL', type=float,
                        help='LM: mu (default: %(default)s)')
    parser.add_argument('--bm25-avg-dl', metavar='REAL', type=float,
                        help='BM25: average DL')
    parser.add_argument('--esa-index-path', metavar='PATH',
                        help='ESA: path to the Indri index')
    parser.add_argument('--esa-k', metavar='K', type=int,
                        help='ESA: depth of ESA vectors (default: %(default)s)')
    parser.add_argument('--esa-threads', metavar='N', type=int,
                        help='ESA: use N threads for retrieval (default: %(default)s)')
    parser.add_argument('--word2vec-model', metavar='FILE',
                        help='Word2Vec: binary model file')
    parser.add_argument('--tagme-token', metavar='TOKEN',
                        help='TAGME: api token')
    parser.add_argument('--tagme-threads', metavar='N', type=int,
                        help='TAGME: run N concurrent requests (default: %(default)s)')
    parser.add_argument('corpus',
                        help='webap or wikiqa')
    parser.add_argument('files', metavar='file', nargs='+',
                        help='input file')
    parser.set_defaults(stemmer='krovetz',
                        stoplist='stoplist.gz',
                        synonyms='synonyms.gz',
                        freqstats='freqstats.gz',
                        lm_mu=10,
                        bm25_avg_dl=0,
                        esa_k=100,
                        esa_threads=1,
                        tagme_threads=1)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')

    if not args.model_path:
        parser.error('Must specify a model path (-m PATH)')
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if not args.corpus:
        parser.error('Must specify the corpus')

    # initialize task context and related resources (i.e., stemmer/stoplist)
    stemmer = get_stemmer(args.stemmer)
    if not stemmer:
        parser.error('unrecognized stemmer name: {}'.format(args.stemmer))

    stoplist = make_stoplist(smart_open(args.stoplist, 'r'))

    ctx = SentenceSelection(stemmer=stemmer, stoplist=stoplist)

    # point to the appropriate procedure
    procedures = {'wikiqa': run_wikiqa,
                  'trecqa': run_trecqa}

    run = procedures.get(args.corpus, None)
    if not run:
        parser.error('Unrecognized corpus: {}'.format(args.corpus))

    # spawn the pipeline and install features
    pipeline = Pipeline(args.model_path)
    pipeline.set_qrels_interface(QREL)

    pipeline.install_feature('Length', features.Length(), QID_TERMS)
    pipeline.install_feature('ExactMatch', features.ExactMatch(), QID_QTEXT, QID_TEXT)
    pipeline.install_feature('Overlap', features.Overlap(), QID_QSTEMS, QID_STEMS)

    synonyms = features.make_synonyms(smart_open(args.synonyms, 'r'))
    pipeline.install_feature('OverlapSyn', features.OverlapSyn(synonyms), QID_QSTEMS, QID_STEMS)

    freqstats = features.make_freqstats(smart_open(args.freqstats, 'r'))
    lm_feature = features.LM(smoothing=features.LM.Dirichlet,
                             freqstats=freqstats,
                             mu=args.lm_mu)
    pipeline.install_feature('LM', lm_feature, QID_QSTEMS, QID_STEMS)

    bm25_feature = features.BM25(freqstats=freqstats,
                                 k1=0.9,
                                 b=0.4,
                                 avg_dl=args.bm25_avg_dl)
    pipeline.install_feature('BM25', bm25_feature, QID_QSTEMS, QID_STEMS)

    esa_feature = features.ZeroFeature
    if args.esa_index_path:
        esa_feature = features.ESACosineSimilarity(index_path=args.esa_index_path,
                                                   k=args.esa_k,
                                                   threads=args.esa_threads,
                                                   batch_size=args.esa_threads * 50,
                                                   cache=(os.path.join(args.model_path, 'cache.QESA.gz'),
                                                          os.path.join(args.model_path, 'cache.ESA.gz')))
    pipeline.install_feature('ESA', esa_feature, QID_QTEXT, QID_TEXT)

    tagme_feature = features.ZeroFeature
    if args.tagme_token:
        tagme_feature = features.TAGMEOverlap(service_uri=features.TAGME_TAG_URI,
                                              token=args.tagme_token,
                                              threads=args.tagme_threads,
                                              batch_size=args.tagme_threads * 2,
                                              cache=(os.path.join(args.model_path, 'cache.QTAGME.gz'),
                                                     os.path.join(args.model_path, 'cache.TAGME.gz')))
    pipeline.install_feature('TAGME', tagme_feature, QID_QTEXT, QID_TEXT)

    if args.word2vec_model:
        logging.info('Load binary Word2Vec model into memory (may take a while...)')
        w2v = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_model, binary=True)
        w2v.init_sims(replace=True)
        pipeline.install_feature('Word2Vec', features.Word2Vec(w2v), QID_QTERMS, QID_TERMS)

    pipeline.install_feature('CPW', features.CPW(), TEXT)
    pipeline.install_feature('SPW', features.SPW(), TEXT)
    pipeline.install_feature('WPS', features.WPS(), TEXT)
    pipeline.install_feature('CWPS', features.WPS(), TEXT)
    pipeline.install_feature('CWR', features.CWR(), TEXT)
    pipeline.install_feature('LWPS', features.LWPS(), TEXT)
    pipeline.install_feature('LWR', features.LWR(), TEXT)
    pipeline.install_feature('DaleChall', features.DaleChall(), TEXT)

    if args.word2vec_model:
        for k, n in [(2, 2), (2, 3), (3, 2), (3, 3)]:
            pipeline.install_feature('MaxScoringNGram[k={},n={}]'.format(k, n),
                                     features.MaxScoringNGram(w2v, k, n), QID_QTERMS, QID_TERMS)

    # run the context
    run(ctx, pipeline, args.files, overwrite=args.overwrite)
