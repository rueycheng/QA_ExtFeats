"""
Features
"""
import collections
import gensim
import glove
import itertools
import json
import math
import logging
import nltk
import numpy as np
import os
import pandas
import re
import sh
import string
import sys
import tempfile
import threading
import urllib

from more_itertools import chunked, pairwise
from queue import Queue
from smart_open import smart_open
from tqdm import tqdm


def CosineSimilarity(qrep, rep, dot, is_unit=False):
    """Yield a sequence of cosine similarity values between query and text vectors.

    Args:
        qrep: a sequence of (qid, qvec) pairs
        rep: a sequence of (qid, vec) pairs
        dot: the `dot product` function
        is_unit: set True if the input are unit vectors
    """

    qvecs = dict(qrep)
    if is_unit:
        for qid, vec in rep:
            yield float(dot(vec, qvecs[qid]))
    else:
        qnorms = {qid: math.sqrt(dot(qvec, qvec)) for qid, qvec in qvecs.items()}
        for qid, vec in rep:
            norm = math.sqrt(dot(vec, vec))
            if norm == 0 or qnorms[qid] == 0:
                yield 0
            else:
                yield float(dot(vec, qvecs[qid])) / norm / qnorms[qid]


def JaccardCoefficient(qrep, rep):
    """Yield a sequence of Jaccard coefficients between query and text tokens.

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """
    q = {qid: frozenset(qtokens) for qid, qtokens in qrep}
    for qid, tokens in rep:
        t = frozenset(tokens)
        if not q[qid] or not t:
            yield 0
        else:
            yield float(len(q[qid] & t)) / len(q[qid] | t)


def OverlapMeasure(qrep, rep):
    """Yield a sequence of set overlap ratios between query and text vectors/sets.

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """
    q = {qid: frozenset(qtokens) for qid, qtokens in qrep}
    qlens = {qid: len(qtokens) for qid, qtokens in q.items()}
    for qid, tokens in rep:
        t = frozenset(tokens)
        if qlens[qid] == 0:
            yield 0
        else:
            yield float(len(q[qid] & t)) / qlens[qid]


class Cache():

    def __init__(self, filepath):
        self.nrows = sum(1 for _ in smart_open(filepath, 'r')) if os.path.exists(filepath) else 0
        self.filepath = filepath

    def __call__(self, rep, transform):
        requested = sum(1 for _ in itertools.islice(rep, self.nrows))
        if requested > 0:
            with smart_open(self.filepath, 'r') as in_:
                for line in itertools.islice(in_, requested):
                    yield eval(line.rstrip())

        if requested == self.nrows:
            with smart_open(self.filepath, 'a') as out_:
                for result in transform(rep):
                    out_.write(str(result) + '\n')
                    yield result


def transform(rep, func, cache=None):

    def _transform(rep):
        rep1, rep2 = itertools.tee(rep)
        return zip((qid for qid, _ in rep1), func(data for _, data in rep2))

    if cache:
        return cache(rep, transform=_transform)
    else:
        return _transform(rep)


# Null features
#
def ZeroFeature(qrep, rep):
    for _ in rep:
        yield 0


# Query-biased summarization features (Metzler and Kanungo, 2008)
#
class Length():
    """Number of tokens in the representation

    Args:
        rep: a sequence of tokens
    """

    def __call__(self, rep):
        for _, tokens in rep:
            yield len(tokens)


class Location():
    """Relative position of the text in a larger text unit, i.e., document.

    Args:
        tags: a sequence of group tags
    """

    def __call__(self, tags):
        for _, group in itertools.groupby(tags):
            size = len(list(group))  # the fastest approach
            for frac in np.linspace(0.0, 1.0, size + 1)[1:]:
                yield frac


class ExactMatch():
    """Binary feature of whether query is a (case-insensitive) substring of the text

    Args:
        qrep: a sequence of (qid, qtext) pairs
        rep: a sequence of (qid, text) pairs
    """

    def __call__(self, qrep, rep):
        qtexts = {qid: qtext.lower() for qid, qtext in qrep}
        for qid, text in rep:
            text = text.lower()
            yield int(qtexts[qid] in text)


class Overlap():
    """Fraction of query tokens that also occur in the text

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """

    def __call__(self, qrep, rep):
        return OverlapMeasure(qrep, rep)


class OverlapSyn():
    """Fraction of query tokens that either occur or have a synonym in the text

    Attributes:
        synonyms: a function that takes a word as input and returns its synonyms

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """

    def __init__(self, synonyms):
        self.synonyms = synonyms

    def __call__(self, qrep, rep):
        qtokens = {qid: [frozenset(self.synonyms(qtoken)) for qtoken in qtokens]
                   for qid, qtokens in qrep}
        for qid, tokens in rep:
            tokens = frozenset(tokens)
            overlap = sum(len(tokens & qset) > 0 for qset in qtokens[qid])
            if not qtokens[qid]:
                yield 0
            else:
                yield float(overlap) / len(qtokens[qid])


def make_synonyms(iterable):
    """A factory method for making WordNet-based synonyms functors"""

    synonyms = {}
    if iterable:
        for line in tqdm(iterable, desc='Load synonyms data', unit=''):
            tokens = line.split()
            synonyms[tokens[0]] = set(tokens[1:])

    def _synonyms(t, inclusive=True):
        res = synonyms.get(t, set())
        return res.union([t]) if inclusive else res
    return _synonyms


class LM():
    """Query likelihood of the sentence language model

    Attributes:
        smoothing: smoothing function
        freqstats: freqstats function
        *kwargs: smoothing method-specific params (e.g., mu, alpha)

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """

    def __init__(self, smoothing, freqstats, **kwargs):
        self.smoothing = smoothing
        self.freqstats = freqstats
        self.params = kwargs

    def __call__(self, qrep, rep):
        return self.smoothing(qrep, rep, self.freqstats, **self.params)

    @staticmethod
    def Dirichlet(qrep, rep, freqstats, mu):
        assert mu > 0
        clen, _ = freqstats(None)
        qtokens = {qid: qtokens for qid, qtokens in qrep}
        for qid, tokens in rep:
            text_tf = collections.Counter(tokens)
            text_len = len(tokens)
            score = 0
            for qtoken in qtokens[qid]:
                ctf, _ = freqstats(qtoken)
                if ctf == 0:
                    continue
                score += math.log(
                    float(text_tf[qtoken] + mu * float(ctf) / clen) / (text_len + mu))
            yield score

    @staticmethod
    def JelinekMercer(qrep, rep, freqstats, alpha):
        assert 0 <= alpha <= 1
        clen, _ = freqstats(None)
        qtokens = {qid: qtokens for qid, qtokens in qrep}
        for qid, tokens in rep:
            text_tf = collections.Counter(tokens)
            text_len = len(tokens)
            score = 0
            for qtoken in qtokens[qid]:
                ctf, _ = freqstats(qtoken)
                if ctf == 0:
                    continue
                score += math.log(
                    (1 - alpha) * float(text_tf[qtoken]) / text_len + alpha * float(ctf) / clen)
            yield score


class BM25():
    """BM25 score function for sentence retrieval

    Attributes:
        freqstats: freqstats function
        k1: param `k1`
        b: param `b`
        avg_dl: param `avg_fl`, estimated automatically if set to 0

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """

    def __init__(self, freqstats, k1, b, avg_dl=0):
        self.freqstats = freqstats
        self.k1 = float(k1)
        self.b = float(b)
        self.avg_dl = float(avg_dl)

    def __call__(self, qrep, rep):
        if self.avg_dl:
            return BM25.BM25(qrep, rep, self.freqstats, self.k1, self.b, self.avg_dl)
        else:
            rep = list(rep)  # this only works when `rep` can be passed on to BM25() later
            counts = [len(tokens) for _, tokens in rep]
            avg_dl = float(sum(counts)) / len(counts) if counts else 0

            logging.info('Set avg_dl automatically to %s' % avg_dl)
            return BM25.BM25(qrep, rep, self.freqstats, self.k1, self.b, avg_dl)

    @staticmethod
    def BM25(qrep, rep, freqstats, k1, b, avg_dl):
        clen, N = freqstats(None)
        qtokens = {qid: qtokens for qid, qtokens in qrep}
        for qid, tokens in rep:
            text_tf = collections.Counter(tokens)
            text_len = len(tokens)
            score = 0
            for qtoken in qtokens[qid]:
                if text_tf[qtoken] > 0:
                    _, df = freqstats(qtoken)
                    idf = math.log((N - df + 0.5) / (df + 0.5))
                    score += idf * (text_tf[qtoken] * (k1 + 1)) / (
                        text_tf[qtoken] + k1 * (1 - b + b * text_len / avg_dl))
            yield score


def make_freqstats(iterable):
    freqstats = {}
    if iterable:
        with tqdm(desc='Load freqstats data (2 steps)') as pbar:
            df = pandas.read_csv(iterable, delim_whitespace=True, names=('term', 'cf', 'df'))
            pbar.update()

            freqstats[None] = (df.cf[0], df.df[0])
            freqstats.update(zip(df.term[1:], zip(df.cf[1:], df.df[1:])))
            del df
            pbar.update()

    def _freqstats(token):
        return freqstats.get(token, (0, 0))
    return _freqstats


class NumDistinctTerms():
    """Number of distinct tokens

    Args:
        rep: a sequence of (qid, tokens) pairs
    """

    def __call__(self, rep):
        for _, tokens in rep:
            yield len(frozenset(tokens))


class NumMatches():
    """Number of matched tokens

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """

    def __call__(self, qrep, rep):
        q = {qid: frozenset(qtokens) for qid, qtokens in qrep}
        for qid, tokens in rep:
            yield sum(1 for t in tokens if t in q[qid])


class MatchingRatio():
    """Number of matched tokens

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """

    def __call__(self, qrep, rep):
        q = {qid: frozenset(qtokens) for qid, qtokens in qrep}
        for qid, tokens in rep:
            if len(tokens) == 0:
                yield 0
            else:
                yield float(sum(1 for t in tokens if t in q[qid])) / len(tokens)


class DistanceBetweenMatches():
    """Maximum distance between any two matched tokens

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """
    def __init__(self, aggregate):
        self.aggregate = aggregate

    def __call__(self, qrep, rep):
        q = {qid: frozenset(qtokens) for qid, qtokens in qrep}
        for qid, tokens in rep:
            indexes = [i for i, t in enumerate(tokens) if t in q[qid]]
            if len(indexes) < 2:
                yield 0
            else:
                yield self.aggregate(indexes)

    @staticmethod
    def Min(indexes):
        return min(b - a for a, b in pairwise(indexes))

    @staticmethod
    def Max(indexes):
        return indexes[-1] - indexes[0]

    @staticmethod
    def Avg(indexes):
        n = float(len(indexes))
        A = sum(i * v for i, v in enumerate(indexes, 1))
        B = sum(indexes)
        return (4.0 / n / (n - 1)) * A + (2.0 * (n + 1) / n / (n - 1)) * B


class NumCapChars():
    """Number of uppercase letters

    Args:
        rep: a sequence of texts
    """

    def __call__(self, rep):
        for text in rep:
            yield sum(1 for c in text if 'A' <= c <= 'Z')


class NumDigitChars():
    """Number of digit characters

    Args:
        rep: a sequence of texts
    """

    def __call__(self, rep):
        for text in rep:
            yield sum(1 for c in text if '0' <= c <= '9')


class NumPunctChars():
    """Number of punctuation characters

    Args:
        rep: a sequence of texts
    """

    def __call__(self, rep):
        punct = frozenset(string.punctuation)
        for text in rep:
            yield sum(1 for c in text if c in punct)


# Embedding based features
#
class Word2Vec():
    """Average cosine similarity between query and text word vectors (Yang et al., 2016)

    Attributes:
        word2vec: a gensim.models.Word2Vec object

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """

    def __init__(self, word2vec):
        assert isinstance(word2vec, gensim.models.KeyedVectors)
        self.word2vec = word2vec

    def __call__(self, qrep, rep):
        qtokens = {qid: [t for t in qtokens if t in self.word2vec] for qid, qtokens in qrep}
        for qid, tokens in rep:
            tokens = [t for t in tokens if t in self.word2vec]
            if qtokens[qid] and tokens:
                yield self.word2vec.n_similarity(qtokens[qid], tokens)
            else:
                yield 0


class Glove():
    """Average cosine similarity between query and text word vectors (Glove)

    Attributes:
        glove: a glove.Glove object (https://github.com/maciejkula/glove-python)
    """

    def __init__(self, glove_):
        assert isinstance(glove_, glove.Glove)
        self.glove = glove_

    def __call__(self, qrep, rep):
        vocab = self.glove.dictionary
        vectors = self.glove.word_vectors
        nil = np.zeros_like(vectors[0])

        qvecs = {}
        for qid, qtokens in qrep:
            qv = np.mean([nil] + [vectors[vocab[t]] for t in qtokens if t in vocab], axis=0)
            qvnorm = np.dot(qv, qv)
            qvecs[qid] = nil if qvnorm == 0 else qv / qvnorm
        for qid, tokens in rep:
            v = np.mean([nil] + [vectors[vocab[t]] for t in tokens if t in vocab], axis=0)
            vnorm = np.dot(v, v)
            vec = nil if vnorm == 0 else v / vnorm
            yield np.dot(qvecs[qid], vec)


# ESA features (Chen et al., 2015; Yang et al., 2016)
#
class ESACosineSimilarity():
    """Cosine similarity between query and text ESA vectors (Chen et al., 2015; Yang et al., 2016)

    Args:
        qrep: a sequence of (qid, qtext) pairs
        rep: a sequence of (qid, text) pairs
    """
    def __init__(self, *args, **kwargs):
        self.esa = lambda texts: get_ESA_vectors(texts, *args, **kwargs)
        self.cache = [Cache(x) if x else None for x in kwargs.pop('cache', (None, None))]

    def __call__(self, qrep, rep):
        return CosineSimilarity(transform(qrep, self.esa, cache=self.cache[0]),
                                transform(rep, self.esa, cache=self.cache[1]),
                                dot=ESACosineSimilarity.dot_over_logarithms)

    @staticmethod
    def dot_over_logarithms(v1, v2):
        """Compute dot product over logarithmic weights"""
        if v1 == v2:
            return sum(math.exp(val + val) for val in v1.values())
        if len(v1) == 0 or len(v2) == 0:
            return 0
        keys = set(v1.keys()) & set(v2.keys())
        return sum(math.exp(v1[k] + v2[k]) for k in keys)


def make_indri_term_trans():
    ascii_chars = set(map(chr, range(0x00, 0x80)))
    mask = ''.join(ascii_chars - set(string.ascii_letters) - set(string.digits))
    # return string.maketrans(mask, ' ' * len(mask))
    return str.maketrans(mask, ' ' * len(mask))


def parse_components(iterable, prefix):
    """Yield ESA vector components from TREC-format retrieval runs."""
    offset = len(prefix)
    for line in iterable:
        if line.startswith('#') or line.startswith('\t'):
            print(line, end='', file=sys.stderr)
            continue
        qid, _, docno, _, sim, _ = line.split()
        if not docno.startswith(prefix):
            continue
        yield {'qid': int(qid), 'wiki_id': int(docno[offset:]), 'score': float(sim)}


def parse_ESA_vectors(iterable, prefix):
    """Yield ESA vectors from TREC-format retrieval runs."""
    for qid, grp in itertools.groupby(parse_components(iterable, prefix), lambda x: x['qid']):
        yield qid, {r['wiki_id']: r['score'] for r in grp}


def get_ESA_vectors(texts,index_path, k, threads=1, batch_size=1000,
                    prefix='ENWIKI-', indri_run_query_cmd='IndriRunQuery'):
    """Yield ESA vectors by querying Indri index.

    Args:
        texts: a sequence of input texts
        index_path: path to the Indri index
        k: number of wikipages to retrieve
        threads: number of threads to use
        batch_size: number of queries in a batch
        prefix: prefix of indexed wikipage docno
        indri_run_query_cmd: path to the executable 'IndriRunQuery'
    """
    assert texts
    assert os.path.exists(index_path)
    assert k > 0

    indri_term_trans = make_indri_term_trans()
    indri_run_query = sh.Command(indri_run_query_cmd)

    for batch in chunked(texts, batch_size):
        with tempfile.NamedTemporaryFile('w', delete=True) as out:
            out.write('<parameters>\n')
            for qid, text in enumerate(batch, 1):
                # ascii_text = text.encode('ascii', errors='replace').replace('?', ' ').lower()
                ascii_text = ''.join([c if ord(c) < 128 else ' ' for c in text]).lower()
                terms = ascii_text.translate(indri_term_trans).split()
                if not terms:
                    continue
                query = '#combine({})'.format(' '.join(terms))
                out.write(
                    '  <query>\n'
                    '    <number>{qid}</number>\n'
                    '    <text>{query}</text>\n'
                    '  </query>\n'
                    .format(qid=qid, query=query)
                )
            out.write('</parameters>\n')
            out.flush()

            # qid's current value is the size of the batch
            all_qids = range(1, qid + 1)

            query_filename = out.name
            runs_input = indri_run_query('-index={}'.format(index_path),
                                         '-count={}'.format(k),
                                         '-threads={}'.format(threads),
                                         '-trecFormat=1',
                                         query_filename,
                                         _iter=True)

            vectors = {qid: vec for qid, vec in parse_ESA_vectors(runs_input, prefix)}
            for qid in all_qids:
                yield vectors.get(qid, {})


# TAGME features (Yang et al., 2016)
#
class TAGMEOverlap():
    """Jaccard coefficient between query and text TAGME entity sets (Yang et al., 2016)

    Args:
        qrep: a sequence of (qid, qtext) pairs
        rep: a sequence of (qid, text) pairs
    """
    def __init__(self, *args, **kwargs):
        self.tagme = lambda texts: get_TAGME_vectors(texts, *args, **kwargs)
        self.cache = [Cache(x) if x else None for x in kwargs.pop('cache', (None, None))]

    def __call__(self, qrep, rep):
        return JaccardCoefficient(transform(qrep, self.tagme, cache=self.cache[0]),
                                  transform(rep, self.tagme, cache=self.cache[1]))


TAGME_TAG_URI = 'https://tagme.d4science.org/tagme/tag'


def parse_TAGME_vectors(iterable):
    """Yield TAGME vectors by parsing raw service output."""
    for rep in iterable:
        data = json.loads(rep)
        annotations = data.get('annotations', [])
        yield {r['id']: float(r['rho']) for r in annotations}


def get_TAGME_vectors(texts, service_uri, token, threads=1, batch_size=10, **kwargs):
    """Yield TAGME vectors by querying a remote TAGME service.

    Args:
        texts: a sequence of input texts
        service_uri: TAGME service uri
        token: service authorization token
        threads: number of threads to use
        batch_size: number of queries in a batch
    """
    assert texts
    assert service_uri
    assert token

    queue = Queue()
    results = None

    def fetcher():
        while True:
            i, data = queue.get()
            try:
                request = urllib.request.Request(service_uri, data.encode('utf-8'))
                rep = urllib.request.urlopen(request).read()
            except Exception as e:
                rep = json.dumps({'error': str(e)})
                print(e, i, data, file=sys.stderr)
            results.append((i, rep))
            queue.task_done()

    for i in range(threads):
        t = threading.Thread(target=fetcher)
        t.daemon = True
        t.start()

    try:
        results = []
        for batch in chunked(texts, batch_size):
            for i, text in enumerate(batch, 1):
                if text == '' or text.isspace():
                    results.append((i, '{}'))
                    continue
                param = {'text': text.encode('utf8', errors='ignore'), 'gcube-token': token}
                for attr in ('lang', 'tweet', 'include_abstract', 'include_categories',
                             'include_all_spots', 'long_text', 'epsilon'):
                    if attr in kwargs and kwargs[attr] is not None:
                        val = kwargs[attr]
                        param[attr] = str(val).lower() if isinstance(val, bool) else str(val)
                data = urllib.parse.urlencode(param)
                queue.put((i, data))
            queue.join()

            results.sort(key=lambda x: x[0])
            for vec in parse_TAGME_vectors(rep for _, rep in results):
                yield vec
    except KeyboardInterrupt:
        sys.exit(1)


# Readability features
#
# PUNCTUATION_REMOVER = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
PUNCTUATION_REMOVER = str.maketrans(string.punctuation, ' ' * len(string.punctuation))


def _tokenize(text):
    # cleaned = text.encode('utf8', errors='replace').translate(PUNCTUATION_REMOVER)
    # return cleaned.decode('utf8').split()
    return text.translate(PUNCTUATION_REMOVER).split()


class SyllableCounter():
    """Fallback syllable counter.

    This is based on the algorithm in Greg Fast's perl module Lingua::EN::Syllable.
    """

    def __init__(self):
        specialSyllables_en = """tottered 2
                                 chummed 1
                                 peeped 1
                                 moustaches 2
                                 shamefully 3
                                 messieurs 2
                                 satiated 4
                                 sailmaker 4
                                 sheered 1
                                 disinterred 3
                                 propitiatory 6
                                 bepatched 2
                                 particularized 5
                                 caressed 2
                                 trespassed 2
                                 sepulchre 3
                                 flapped 1
                                 hemispheres 3
                                 pencilled 2
                                 motioned 2
                                 poleman 2
                                 slandered 2
                                 sombre 2
                                 etc 4
                                 sidespring 2
                                 mimes 1
                                 effaces 2
                                 mr 2
                                 mrs 2
                                 ms 1
                                 dr 2
                                 st 1
                                 sr 2
                                 jr 2
                                 truckle 2
                                 foamed 1
                                 fringed 2
                                 clattered 2
                                 capered 2
                                 mangroves 2
                                 suavely 2
                                 reclined 2
                                 brutes 1
                                 effaced 2
                                 quivered 2
                                 h'm 1
                                 veriest 3
                                 sententiously 4
                                 deafened 2
                                 manoeuvred 3
                                 unstained 2
                                 gaped 1
                                 stammered 2
                                 shivered 2
                                 discoloured 3
                                 gravesend 2
                                 60 2
                                 lb 1
                                 unexpressed 3
                                 greyish 2
                                 unostentatious 5
        """

        fallback_cache = {}

        fallback_subsyl = ["cial", "tia", "cius", "cious", "gui", "ion", "iou",
                           "sia$", ".ely$"]

        fallback_addsyl = ["ia", "riet", "dien", "iu", "io", "ii",
                           "[aeiouy]bl$", "mbl$",
                           "[aeiou]{3}",
                           "^mc", "ism$",
                           "(.)(?!\\1)([aeiouy])\\2l$",
                           "[^l]llien",
                           "^coad.", "^coag.", "^coal.", "^coax.",
                           "(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]",
                           "dnt$"]

        # Compile our regular expressions
        for i in range(len(fallback_subsyl)):
            fallback_subsyl[i] = re.compile(fallback_subsyl[i])
        for i in range(len(fallback_addsyl)):
            fallback_addsyl[i] = re.compile(fallback_addsyl[i])

        # Read our syllable override file and stash that info in the cache
        for line in specialSyllables_en.splitlines():
            line = line.strip()
            if line:
                toks = line.split()
                assert len(toks) == 2
                fallback_cache[toks[0].strip().lower()] = int(toks[1])

        self.fallback_cache = fallback_cache
        self.fallback_subsyl = fallback_subsyl
        self.fallback_addsyl = fallback_addsyl

    def __call__(self, word):
        word = word.strip().lower()
        if not word:
            return 0

        # Check for a cached syllable count
        count = self.fallback_cache.get(word, -1)
        if count > 0:
            return count

        # Remove final silent 'e'
        if word[-1] == "e":
            word = word[:-1]

        # Count vowel groups
        count = 0
        prev_was_vowel = 0
        for c in word:
            is_vowel = c in ("a", "e", "i", "o", "u", "y")
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Add & subtract syllables
        for r in self.fallback_addsyl:
            if r.search(word):
                count += 1
        for r in self.fallback_subsyl:
            if r.search(word):
                count -= 1

        # Cache the syllable count
        self.fallback_cache[word] = count
        return count


def analyze(texts, paragraph=True):
    syllable_counter = SyllableCounter()
    if paragraph:
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for text in texts:
        sentences = sentence_tokenizer.tokenize(text) if paragraph else [text]
        sentence_count = len(sentences)

        words = _tokenize(text)
        word_count = len(words)
        char_count = sum(len(word) for word in words)
        syllable_counts = [syllable_counter(word) for word in words]

        long_word_count = sum(len(word) >= 7 for word in words)
        complex_word_count = 0
        for word, syllable_count in zip(words, syllable_counts):
            if syllable_count >= 3:
                if word[0].islower() or any(sent.startswith(word) for sent in sentences):
                    print(word, syllable_count)
                    complex_word_count += 1

        yield {
            'words': words,
            'word_cnt': float(word_count),
            'char_cnt': float(char_count),
            'syllable_cnt': float(sum(syllable_counts)),
            'sentence_cnt': float(sentence_count),
            'long_word_cnt': float(long_word_count),
            'complex_word_cnt': float(complex_word_count),
            'avg_words_p_sentence': float(word_count) / sentence_count,
        }


# Surface features
#
class CPW():
    """Number of characters per word."""

    def __call__(self, rep):
        for m in analyze(rep):
            yield m['char_cnt'] / m['word_cnt'] if m['word_cnt'] > 0 else 0


class SPW():
    """Number of syllables per word."""

    def __call__(self, rep):
        for m in analyze(rep):
            yield m['syllable_cnt'] / m['word_cnt'] if m['word_cnt'] > 0 else 0


class WPS():
    """Number of words per sentence."""

    def __call__(self, rep):
        for m in analyze(rep):
            yield m['avg_words_p_sentence']


class CWPS():
    """Number of complex words per sentence."""

    def __call__(self, rep):
        for m in analyze(rep):
            yield m['complex_word_cnt'] / m['sentence_cnt']


class CWR():
    """Ratio of complex words to words."""

    def __call__(self, rep):
        for m in analyze(rep):
            yield m['complex_word_cnt'] / m['word_cnt'] if m['word_cnt'] > 0 else 0


class LWPS():
    """Number of long words per sentence.

    This feature is nearly identical to RIX.
    """

    def __call__(self, rep):
        for m in analyze(rep):
            yield m['long_word_cnt'] / m['sentence_cnt'] if m['word_cnt'] > 0 else 0


class LWR():
    """Ratio of long words to words."""

    def __call__(self, rep):
        for m in analyze(rep):
            yield m['long_word_cnt'] / m['word_cnt'] if m['word_cnt'] > 0 else 0


# Readability indexes
#
class ARI():

    def __call__(self, rep):
        for m in analyze(rep):
            score = 0.0
            if m['word_cnt'] > 0:
                score = 4.71 * (m['char_cnt'] / m['word_cnt']) + 0.5 * m['avg_words_p_sentence'] - 21.43
            yield score


class FleschReadingEase():

    def __call__(self, rep):
        for m in analyze(rep):
            score = 0.0
            if m['word_cnt'] > 0.0:
                score = 206.835 - (1.015 * (m['avg_words_p_sentence'])) - (84.6 * (m['syllable_cnt'] / m['word_cnt']))
            yield round(score, 4)


class FleschKincaidGradeLevel():

    def __call__(self, rep):
        for m in analyze(rep):
            score = 0.0
            if m['word_cnt'] > 0.0:
                score = 0.39 * (m['avg_words_p_sentence']) + 11.8 * (m['syllable_cnt'] / m['word_cnt']) - 15.59
            yield round(score, 4)


class GunningFogIndex():

    def __call__(self, rep):
        for m in analyze(rep):
            score = 0.0
            if m['word_cnt'] > 0.0:
                score = 0.4 * m['avg_words_p_sentence'] + 40.0 * m['complex_word_cnt'] / m['word_cnt']
            yield round(score, 4)


class SMOGIndex():

    def __call__(self, rep):
        for m in analyze(rep):
            score = 0.0
            if m['word_cnt'] > 0.0:
                #  score = math.sqrt(m['complex_word_cnt'] * (30 / m['sentence_cnt'])) + 3
                score = 3 + math.sqrt(30.0 * m['complex_word_cnt'] / m['sentence_cnt'])
            yield score


class ColemanLiauIndex():

    def __call__(self, rep):
        for m in analyze(rep):
            score = 0.0
            if m['word_cnt'] > 0.0:
                score = 5.89 * (m['char_cnt'] / m['word_cnt']) - 30.0 * (m['sentence_cnt'] / m['word_cnt']) - 15.8
            yield round(score, 4)


class LIX():

    def __call__(self, rep):
        for m in analyze(rep):
            score = 0.0
            if m['word_cnt'] > 0.0:
                #  score = m['word_cnt'] / m['sentence_cnt'] + float(100 * longwords) / m['word_cnt']
                score = m['word_cnt'] / m['sentence_cnt'] + 100.0 * m['long_word_cnt'] / m['word_cnt']
            yield score


class RIX():

    def __call__(self, rep):
        for m in analyze(rep):
            score = 0.0
            if m['word_cnt'] > 0.0:
                score = m['long_word_cnt'] / m['sentence_cnt']
            yield score


class DaleChall():

    FAMILIAR_WORDS = set(r'''
            America American April August Christmas December English February
            French Friday God I I'd I'll I'm I've Indian January July June March
            May Miss Monday Mr.  Mrs.  Negro November October Saturday September
            States Sunday Thanksgiving Thursday Tuesday United Wednesday a able
            aboard about above absent accept accident account ache aching acorn
            acre across act acts add address admire adventure afar afraid after
            afternoon afterward afterwards again against age aged ago agree ah
            ahead aid aim air airfield airplane airport airship airy alarm alike
            alive all alley alligator allow almost alone along aloud already also
            always am among amount an and angel anger angry animal another answer
            ant any anybody anyhow anyone anything anyway anywhere apart apartment
            ape apiece appear apple apron are aren't arise arithmetic arm armful
            army arose around arrange arrive arrived arrow art artist as ash ashes
            aside ask asleep at ate attack attend attention aunt author auto
            automobile autumn avenue awake awaken away awful awfully awhile ax axe
            baa babe babies back background backward backwards bacon bad badge
            badly bag bake baker bakery baking ball balloon banana band bandage
            bang banjo bank banker bar barber bare barefoot barely bark barn barrel
            base baseball basement basket bat batch bath bathe bathing bathroom
            bathtub battle battleship bay be beach bead beam bean bear beard beast
            beat beating beautiful beautify beauty became because become becoming
            bed bedbug bedroom bedspread bedtime bee beech beef beefsteak beehive
            been beer beet before beg began beggar begged begin beginning begun
            behave behind being believe bell belong below belt bench bend beneath
            bent berries berry beside besides best bet better between bib bible
            bicycle bid big bigger bill billboard bin bind bird birth birthday
            biscuit bit bite biting bitter black blackberry blackbird blackboard
            blackness blacksmith blame blank blanket blast blaze bleed bless
            blessing blew blind blindfold blinds block blood bloom blossom blot
            blow blue blueberry bluebird blush board boast boat bob bobwhite bodies
            body boil boiler bold bone bonnet boo book bookcase bookkeeper boom
            boot born borrow boss both bother bottle bottom bought bounce bow
            bow-wow bowl box boxcar boxer boxes boy boyhood bracelet brain brake
            bran branch brass brave bread break breakfast breast breath breathe
            breeze brick bride bridge bright brightness bring broad broadcast broke
            broken brook broom brother brought brown brush bubble bucket buckle bud
            buffalo bug buggy build building built bulb bull bullet bum bumblebee
            bump bun bunch bundle bunny burn burst bury bus bush bushel business
            busy but butcher butt butter buttercup butterfly buttermilk
            butterscotch button buttonhole buy buzz by bye cab cabbage cabin
            cabinet cackle cage cake calendar calf call caller calling came camel
            camp campfire can can't canal canary candle candlestick candy cane
            cannon cannot canoe canyon cap cape capital captain car card cardboard
            care careful careless carelessness carload carpenter carpet carriage
            carrot carry cart carve case cash cashier castle cat catbird catch
            catcher caterpillar catfish catsup cattle caught cause cave ceiling
            cell cellar cent center cereal certain certainly chain chair chalk
            champion chance change chap charge charm chart chase chatter cheap
            cheat check checkers cheek cheer cheese cherry chest chew chick chicken
            chief child childhood children chill chilly chimney chin china chip
            chipmunk chocolate choice choose chop chorus chose chosen christen
            church churn cigarette circle circus citizen city clang clap class
            classmate classroom claw clay clean cleaner clear clerk clever click
            cliff climb clip cloak clock close closet cloth clothes clothing cloud
            cloudy clover clown club cluck clump coach coal coast coat cob cobbler
            cocoa coconut cocoon cod codfish coffee coffeepot coin cold collar
            college color colored colt column comb come comfort comic coming
            company compare conductor cone connect coo cook cooked cookie cookies
            cooking cool cooler coop copper copy cord cork corn corner correct cost
            cot cottage cotton couch cough could couldn't count counter country
            county course court cousin cover cow coward cowardly cowboy cozy crab
            crack cracker cradle cramps cranberry crank cranky crash crawl crazy
            cream creamy creek creep crept cried cries croak crook crooked crop
            cross cross-eyed crossing crow crowd crowded crown cruel crumb crumble
            crush crust cry cub cuff cuff cup cup cupboard cupful cure curl curly
            curtain curve cushion custard customer cut cute cutting dab dad daddy
            daily dairy daisy dam damage dame damp dance dancer dancing dandy
            danger dangerous dare dark darkness darling darn dart dash date
            daughter dawn day daybreak daytime dead deaf deal dear death decide
            deck deed deep deer defeat defend defense delight den dentist depend
            deposit describe desert deserve desire desk destroy devil dew diamond
            did didn't die died dies difference different dig dim dime dine
            ding-dong dinner dip direct direction dirt dirty discover dish dislike
            dismiss ditch dive diver divide do dock doctor does doesn't dog doll
            dollar dolly don't done donkey door doorbell doorknob doorstep dope dot
            double dough dove down downstairs downtown dozen drag drain drank draw
            draw drawer drawing dream dress dresser dressmaker drew dried drift
            drill drink drip drive driven driver drop drove drown drowsy drub drum
            drunk dry duck due dug dull dumb dump during dust dusty duty dwarf
            dwell dwelt dying each eager eagle ear early earn earth east eastern
            easy eat eaten edge egg eh eight eighteen eighth eighty either elbow
            elder eldest electric electricity elephant eleven elf elm else
            elsewhere empty end ending enemy engine engineer enjoy enough enter
            envelope equal erase eraser errand escape eve even evening ever every
            everybody everyday everyone everything everywhere evil exact except
            exchange excited exciting excuse exit expect explain extra eye eyebrow
            fable face facing fact factory fail faint fair fairy faith fake fall
            false family fan fancy far far-off faraway fare farm farmer farming
            farther fashion fast fasten fat father fault favor favorite fear feast
            feather fed feed feel feet fell fellow felt fence fever few fib fiddle
            field fife fifteen fifth fifty fig fight figure file fill film finally
            find fine finger finish fire firearm firecracker fireplace fireworks
            firing first fish fisherman fist fit fits five fix flag flake flame
            flap flash flashlight flat flea flesh flew flies flight flip flip-flop
            float flock flood floor flop flour flow flower flowery flutter fly foam
            fog foggy fold folks follow following fond food fool foolish foot
            football footprint for forehead forest forget forgive forgot forgotten
            fork form fort forth fortune forty forward fought found fountain four
            fourteen fourth fox frame free freedom freeze freight fresh fret fried
            friend friendly friendship frighten frog from front frost frown froze
            fruit fry fudge fuel full fully fun funny fur furniture further fuzzy
            gain gallon gallop game gang garage garbage garden gas gasoline gate
            gather gave gay gear geese general gentle gentleman gentlemen geography
            get getting giant gift gingerbread girl give given giving glad gladly
            glance glass glasses gleam glide glory glove glow glue go goal goat
            gobble god godmother goes going gold golden goldfish golf gone good
            good-by good-bye good-looking goodbye goodbye goodness goods goody
            goose gooseberry got govern government gown grab gracious grade grain
            grand grandchild grandchildren granddaughter grandfather grandma
            grandmother grandpa grandson grandstand grape grapefruit grapes grass
            grasshopper grateful grave gravel graveyard gravy gray graze grease
            great green greet grew grind groan grocery ground group grove grow
            guard guess guest guide gulf gum gun gunpowder guy ha habit had hadn't
            hail hair haircut hairpin half hall halt ham hammer hand handful
            handkerchief handle handwriting hang happen happily happiness happy
            harbor hard hardly hardship hardware hare hark harm harness harp
            harvest has hasn't haste hasten hasty hat hatch hatchet hate haul have
            haven't having hawk hay hayfield haystack he he'd he'll he's head
            headache heal health healthy heap hear heard hearing heart heat heater
            heaven heavy heel height held hell hello helmet help helper helpful hem
            hen henhouse her herd here here's hero hers herself hey hickory hid
            hidden hide high highway hill hillside hilltop hilly him himself hind
            hint hip hire his hiss history hit hitch hive ho hoe hog hold holder
            hole holiday hollow holy home homely homesick honest honey honeybee
            honeymoon honk honor hood hoof hook hoop hop hope hopeful hopeless horn
            horse horseback horseshoe hose hospital host hot hotel hound hour house
            housetop housewife housework how however howl hug huge hum humble hump
            hundred hung hunger hungry hunk hunt hunter hurrah hurried hurry hurt
            husband hush hut hymn ice icy idea ideal if ill important impossible
            improve in inch inches income indeed indoors ink inn insect inside
            instant instead insult intend interested interesting into invite iron
            is island isn't it it's its itself ivory ivy jacket jacks jail jam jar
            jaw jay jelly jellyfish jerk jig job jockey join joke joking jolly
            journey joy joyful joyous judge jug juice juicy jump junior junk just
            keen keep kept kettle key kick kid kill killed kind kindly kindness
            king kingdom kiss kitchen kite kitten kitty knee kneel knew knife knit
            knives knob knock knot know known lace lad ladder ladies lady laid lake
            lamb lame lamp land lane language lantern lap lard large lash lass last
            late laugh laundry law lawn lawyer lay lazy lead leader leaf leak lean
            leap learn learned least leather leave leaving led left leg lemon
            lemonade lend length less lesson let let's letter letting lettuce level
            liberty library lice lick lid lie life lift light lightness lightning
            like likely liking lily limb lime limp line linen lion lip list listen
            lit little live lively liver lives living lizard load loaf loan loaves
            lock locomotive log lone lonely lonesome long look lookout loop loose
            lord lose loser loss lost lot loud love lovely lover low luck lucky
            lumber lump lunch lying ma machine machinery mad made magazine magic
            maid mail mailbox mailman major make making male mama mamma man manager
            mane manger many map maple marble march mare mark market marriage
            married marry mask mast master mat match matter mattress may maybe
            mayor maypole me meadow meal mean means meant measure meat medicine
            meet meeting melt member men mend meow merry mess message met metal mew
            mice middle midnight might mighty mile miler milk milkman mill million
            mind mine miner mint minute mirror mischief miss misspell mistake misty
            mitt mitten mix moment money monkey month moo moon moonlight moose mop
            more morning morrow moss most mostly mother motor mount mountain mouse
            mouth move movie movies moving mow much mud muddy mug mule multiply
            murder music must my myself nail name nap napkin narrow nasty naughty
            navy near nearby nearly neat neck necktie need needle needn't neighbor
            neighborhood neither nerve nest net never nevermore new news newspaper
            next nibble nice nickel night nightgown nine nineteen ninety no nobody
            nod noise noisy none noon nor north northern nose not note nothing
            notice now nowhere number nurse nut o'clock oak oar oatmeal oats obey
            ocean odd of off offer office officer often oh oil old old-fashioned on
            once one onion only onward open or orange orchard order ore organ other
            otherwise ouch ought our ours ourselves out outdoors outfit outlaw
            outline outside outward oven over overalls overcoat overeat overhead
            overhear overnight overturn owe owing owl own owner ox pa pace pack
            package pad page paid pail pain painful paint painter painting pair pal
            palace pale pan pancake pane pansy pants papa paper parade pardon
            parent park part partly partner party pass passenger past paste pasture
            pat patch path patter pave pavement paw pay payment pea peace peaceful
            peach peaches peak peanut pear pearl peas peck peek peel peep peg pen
            pencil penny people pepper peppermint perfume perhaps person pet phone
            piano pick pickle picnic picture pie piece pig pigeon piggy pile pill
            pillow pin pine pineapple pink pint pipe pistol pit pitch pitcher pity
            place plain plan plane plant plate platform platter play player
            playground playhouse playmate plaything pleasant please pleasure plenty
            plow plug plum pocket pocketbook poem point poison poke pole police
            policeman polish polite pond ponies pony pool poor pop popcorn popped
            porch pork possible post postage postman pot potato potatoes pound pour
            powder power powerful praise pray prayer prepare present pretty price
            prick prince princess print prison prize promise proper protect proud
            prove prune public puddle puff pull pump pumpkin punch punish pup pupil
            puppy pure purple purse push puss pussy pussycat put putting puzzle
            quack quart quarter queen queer question quick quickly quiet quilt quit
            quite rabbit race rack radio radish rag rail railroad railway rain
            rainbow rainy raise raisin rake ram ran ranch rang rap rapidly rat rate
            rather rattle raw ray reach read reader reading ready real really reap
            rear reason rebuild receive recess record red redbird redbreast refuse
            reindeer rejoice remain remember remind remove rent repair repay repeat
            report rest return review reward rib ribbon rice rich rid riddle ride
            rider riding right rim ring rip ripe rise rising river road roadside
            roar roast rob robber robe robin rock rocket rocky rode roll roller
            roof room rooster root rope rose rosebud rot rotten rough round route
            row rowboat royal rub rubbed rubber rubbish rug rule ruler rumble run
            rung runner running rush rust rusty rye sack sad saddle sadness safe
            safety said sail sailboat sailor saint salad sale salt same sand
            sandwich sandy sang sank sap sash sat satin satisfactory sausage savage
            save savings saw say scab scales scare scarf school schoolboy
            schoolhouse schoolmaster schoolroom scorch score scrap scrape scratch
            scream screen screw scrub sea seal seam search season seat second
            secret see seed seeing seek seem seen seesaw select self selfish sell
            send sense sent sentence separate servant serve service set setting
            settle settlement seven seventeen seventh seventy several sew shade
            shadow shady shake shaker shaking shall shame shan't shape share sharp
            shave she she'd she'll she's shear shears shed sheep sheet shelf shell
            shepherd shine shining shiny ship shirt shock shoe shoemaker shone
            shook shoot shop shopping shore short shot should shoulder shouldn't
            shout shovel show shower shut shy sick sickness side sidewalk sideways
            sigh sight sign silence silent silk sill silly silver simple sin since
            sing singer single sink sip sir sis sissy sister sit sitting six
            sixteen sixth sixty size skate skater ski skin skip skirt sky slam slap
            slate slave sled sleep sleepy sleeve sleigh slept slice slid slide
            sling slip slipped slipper slippery slit slow slowly sly smack small
            smart smell smile smoke smooth snail snake snap snapping sneeze snow
            snowball snowflake snowy snuff snug so soak soap sob socks sod soda
            sofa soft soil sold soldier sole some somebody somehow someone
            something sometime sometimes somewhere son song soon sore sorrow sorry
            sort soul sound soup sour south southern space spade spank sparrow
            speak speaker spear speech speed spell spelling spend spent spider
            spike spill spin spinach spirit spit splash spoil spoke spook spoon
            sport spot spread spring springtime sprinkle square squash squeak
            squeeze squirrel stable stack stage stair stall stamp stand star stare
            start starve state station stay steak steal steam steamboat steamer
            steel steep steeple steer stem step stepping stick sticky stiff still
            stillness sting stir stitch stock stocking stole stone stood stool
            stoop stop stopped stopping store stories stork storm stormy story
            stove straight strange stranger strap straw strawberry stream street
            stretch string strip stripes strong stuck study stuff stump stung
            subject such suck sudden suffer sugar suit sum summer sun sunflower
            sung sunk sunlight sunny sunrise sunset sunshine supper suppose sure
            surely surface surprise swallow swam swamp swan swat swear sweat
            sweater sweep sweet sweetheart sweetness swell swept swift swim
            swimming swing switch sword swore table tablecloth tablespoon tablet
            tack tag tail tailor take taken taking tale talk talker tall tame tan
            tank tap tape tar tardy task taste taught tax tea teach teacher team
            tear tease teaspoon teeth telephone tell temper ten tennis tent term
            terrible test than thank thankful thanks that that's the theater thee
            their them then there these they they'd they'll they're they've thick
            thief thimble thin thing think third thirsty thirteen thirty this thorn
            those though thought thousand thread three threw throat throne through
            throw thrown thumb thunder thy tick ticket tickle tie tiger tight till
            time tin tinkle tiny tip tiptoe tire tired title to toad toadstool
            toast tobacco today toe together toilet told tomato tomorrow ton tone
            tongue tonight too took tool toot tooth toothbrush toothpick top tore
            torn toss touch tow toward towards towel tower town toy trace track
            trade train tramp trap tray treasure treat tree trick tricycle tried
            trim trip trolley trouble truck true truly trunk trust truth try tub
            tug tulip tumble tune tunnel turkey turn turtle twelve twenty twice
            twig twin two ugly umbrella uncle under understand underwear undress
            unfair unfinished unfold unfriendly unhappy unhurt uniform unkind
            unknown unless unpleasant until unwilling up upon upper upset upside
            upstairs uptown upward us use used useful valentine valley valuable
            value vase vegetable velvet very vessel victory view village vine
            violet visit visitor voice vote wag wagon waist wait wake waken walk
            wall walnut want war warm warn was wash washer washtub wasn't waste
            watch watchman water watermelon waterproof wave wax way wayside we we'd
            we'll we're we've weak weaken weakness wealth weapon wear weary weather
            weave web wedding wee weed week weep weigh welcome well went were west
            western wet whale what what's wheat wheel when whenever where which
            while whip whipped whirl whiskey whisky whisper whistle white who who'd
            who'll who's whole whom whose why wicked wide wife wiggle wild wildcat
            will willing willow win wind windmill window windy wine wing wink
            winner winter wipe wire wise wish wit witch with without woke wolf
            woman women won won't wonder wonderful wood wooden woodpecker woods
            wool woolen word wore work worker workman world worm worn worry worse
            worst worth would wouldn't wound wove wrap wrapped wreck wren wring
            write writing written wrong wrote wrung yard yarn year yell yellow yes
            yesterday yet yolk yonder you you'd you'll you're you've young
            youngster your yours yourself yourselves youth
    '''.split())

    def __call__(self, rep):
        for m in analyze(rep):
            pdw = float(0)
            if m['word_cnt'] > 0:
                num_dw = len([1 for word in m['words'] if word not in DaleChall.FAMILIAR_WORDS])
                pdw = float(num_dw) / m['word_cnt']

            score = 0.1549 * pdw + 0.0496 * (m['word_cnt'] / m['sentence_cnt'])
            if pdw > 0.05:
                score += 3.6365
            yield score


# Features with strong focus on question answering
#


# Courtesy of https://gist.github.com/hellertime/3060604
def ngrams(xs, f=None, n=3):
    ts = itertools.tee(xs, n)
    for i, t in enumerate(ts[1:]):
        for _ in range(i + 1):
            next(t, None)
    res = zip(*ts)
    return map(f, res) if f is not None else res


class MaxScoringNGram():
    """Average cosine similarity between "query head" and max-scoring n-gram vectors

    Attributes:
        word2vec: a gensim.models.Word2Vec object
        k: use the first `k` query terms as the "query head" (None = use all terms)
        n: window size

    Args:
        qrep: a sequence of (qid, qtokens) pairs
        rep: a sequence of (qid, tokens) pairs
    """

    def __init__(self, word2vec, k, n=1):
        assert isinstance(word2vec, gensim.models.KeyedVectors)
        self.word2vec = word2vec
        self.k = k
        self.n = n

    def __call__(self, qrep, rep):
        qtokens = {qid: [t for t in qtokens[self.k:] if t in self.word2vec]
                   for qid, qtokens in qrep}
        for qid, tokens in rep:
            best_score = 0
            for ngram in ngrams(tokens, n=self.n):
                ngram_tokens = [t for t in ngram if t in self.word2vec]
                if qtokens[qid] and ngram_tokens:
                    score = self.word2vec.n_similarity(qtokens[qid], ngram_tokens)
                else:
                    score = 0
                if score > best_score:
                    best_score = score
            yield best_score
