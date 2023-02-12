from __future__ import print_function

import argparse
import itertools

from smart_open import smart_open


def get_rows(iterable):
    for line in iterable:
        if line.startswith('#'):
            yield '#', None, None, line
            continue
        fields = line.split()
        qid = fields[1][4:]  # qid:XXXX
        docno = fields[-1]
        rel = int(fields[0])
        yield qid, docno, rel, line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input file')
    args = parser.parse_args()

    rows = get_rows(smart_open(args.input_file))
    for k, grp in itertools.groupby(rows, lambda x: x[0]):
        items = list(grp)
        if k != '#':
            if all(item[2] <= 0 for item in items):
                continue
        for _, _, _, line in items:
            print(line, end='')
