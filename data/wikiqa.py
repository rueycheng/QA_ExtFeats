"""
Tools for parsing the WikiQA corpus

Microsoft Research WikiQA Corpus
https://www.microsoft.com/en-us/download/details.aspx?id=52419
"""
import itertools
import more_itertools


def get_rows(iterable):
    """Yield a sequence of rows as dict's

    The input format follows that of WikiQA.tsv
    """
    firstline = next(iterable, None)
    header = firstline.split()
    for line in iterable:
        row = line.rstrip().split('\t')
        assert len(header) == len(row)
        yield dict(zip(header, row))


def get_questions(iterable):
    """Yield a sequence of questions, each with a list of labeled answers.

    The input format follows that of WikiQA.tsv
    """
    for k, grp in itertools.groupby(get_rows(iterable), lambda x: x['QuestionID']):
        rows = list(grp)
        answers = []
        for row in rows:
            answers.append({'sentence_id': row['SentenceID'],
                            'sentence': row['Sentence'],
                            'rel': int(row['Label'])})
        yield {'qid': rows[0]['QuestionID'],
               'question': rows[0]['Question'],
               'doc_id': rows[0]['DocumentID'],
               'doc_title': rows[0]['DocumentTitle'],
               'answers': answers}


def get_topics(iterable):
    for q in more_itertools.unique_justseen((q for q in get_questions(iterable)), key=lambda x: x['qid']):
        topic = {k: q[k] for k in ('qid', 'question')}
        yield topic['question'], topic


def get_sentences_in_queries(iterable):
    for q in get_questions(iterable):
        doc = {k: q[k] for k in ('qid', 'doc_title')}
        doc['docno'] = q['doc_id']
        sentences = [a['sentence'] for a in q['answers']]
        rels = [a['rel'] for a in q['answers']]
        yield sentences, rels, doc
