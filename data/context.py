"""
Context object and representation functions
"""
import string

from tqdm import tqdm


def _load(name, data):
    return list(tqdm(data, desc='Load {}'.format(name), unit=''))


# PUNCTUATION_REMOVER = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
PUNCTUATION_REMOVER = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

def _tokenize(text):
    # return text.encode('utf8', errors='replace').translate(PUNCTUATION_REMOVER).lower().split()
    return text.translate(PUNCTUATION_REMOVER).lower().split()


class SentenceSelection():
    """Context object for hosting data and resources"""

    def __init__(self, stemmer=None, stoplist=None):
        self.stemmer = stemmer if stemmer else lambda x: x  # no stemming by default
        self.stoplist = stoplist if stoplist else lambda x: False  # no stoplist by default

    def load_data(self, topics, sentence_data):
        self.topics = _load('topics', topics)
        self.sentence_data = _load('sentence data', sentence_data)


def QID_QTEXT(ctx):
    for topic, metadata in ctx.topics:
        yield metadata['qid'], topic


def QID_QTOKENS(ctx):
    for qid, qtext in QID_QTEXT(ctx):
        yield qid, [t for t in _tokenize(qtext)]


def QID_QTERMS(ctx):
    for qid, qtext in QID_QTEXT(ctx):
        yield qid, [t for t in _tokenize(qtext) if not ctx.stoplist(t)]


def QID_QSTEMS(ctx):
    for qid, qterms in QID_QTERMS(ctx):
        yield qid, [ctx.stemmer(t) for t in qterms]


def QID_TEXT(ctx):
    for sentences, _, metadata in ctx.sentence_data:
        for sentence in sentences:
            yield metadata['qid'], sentence


def QID_TOKENS(ctx):
    for qid, text in QID_TEXT(ctx):
        yield qid, [t for t in _tokenize(text)]


def QID_TERMS(ctx):
    for qid, text in QID_TEXT(ctx):
        yield qid, [t for t in _tokenize(text) if not ctx.stoplist(t)]


def QID_STEMS(ctx):
    for qid, terms in QID_TERMS(ctx):
        yield qid, [ctx.stemmer(t) for t in terms]


def DOCNO(ctx):
    for sentences, _, metadata in ctx.sentence_data:
        for _ in sentences:
            yield metadata['docno']


def TEXT(ctx):
    for sentences, _, _ in ctx.sentence_data:
        for sentence in sentences:
            yield sentence


def QREL(ctx):
    for sentences, rels, metadata in ctx.sentence_data:
        for i, rel in enumerate(rels, 1):
            yield metadata['qid'], '{}.{}'.format(metadata['docno'], i), rel
