"""
Tools for parsing TREC QA data
"""
import re


def get_qapairs(iterable):
    """Yield a sequence of question-answer objects.

    The input has to be in the XML format as with the NAACL '13 version.
    """
    QAPAIR_PATTERN = re.compile(r"<QApairs id='(\S+)'>")

    buf = None
    for line in iterable:
        if line.startswith('<QApairs '):
            m = re.match(QAPAIR_PATTERN, line)
            qid = m.group(1)
            question = None
            answers = []
        elif line.startswith('</QApairs>'):
            yield {'qid': qid, 'question': question, 'answers': answers}
        elif line.startswith('<question>'):
            buf = []
        elif line.startswith('</question>'):
            question = ' '.join(buf[0].split())
        elif line.startswith('<positive>') or line.startswith('<negative>'):
            buf = []
        elif line.startswith('</positive>'):
            answers.append({'answer': ' '.join(buf[0].split()), 'rel': 1})
        elif line.startswith('</negative>'):
            answers.append({'answer': ' '.join(buf[0].split()), 'rel': 0})
        else:
            buf.append(line)


def get_topics(iterable):
    for q in get_qapairs(iterable):
        topic = {k: q[k] for k in ('qid', 'question')}
        yield topic['question'], topic


def get_sentences_in_queries(iterable):
    for q in get_qapairs(iterable):
        doc = {k: q[k] for k in ('qid',)}
        doc['docno'] = '{}_answers'.format(q['qid'])
        sentences = [a['answer'] for a in q['answers']]
        rels = [a['rel'] for a in q['answers']]
        yield sentences, rels, doc
