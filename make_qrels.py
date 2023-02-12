from __future__ import print_function

import argparse

from smart_open import smart_open


def get_rows(iterable):
    for line in iterable:
        if line.startswith('#'):
            continue
        fields = line.split()
        qid = fields[1][4:]  # qid:XXXX
        docno = fields[-1]
        if docno.startswith('docno:'):
            docno = docno[6:]  # docno:XXXXXXXX
        rel = int(fields[0])
        yield qid, docno, rel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input file')
    parser.add_argument('-c', dest='compact', action='store_true',
                        help='generate compact output')
    args = parser.parse_args()

    rows = get_rows(smart_open(args.input_file))
    if args.compact:
        for qid, docno, rel in rows:
            if rel <= 0:
                continue
            print(qid, '0', docno, rel)
    else:
        for qid, docno, rel in rows:
            print(qid, '0', docno, rel)
