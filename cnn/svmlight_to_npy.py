from __future__ import print_function

import argparse
import numpy as np

from smart_open import smart_open


def parse_svmlight_lines(iterable):
    for line in iterable:
        if line.startswith('#'):
            continue
        head, comment = line.split('#', 1)

        fields = head.split()
        rel = int(fields[0])
        qid = fields[1][4:]  # qid:XXXX

        vector = {}
        for f in fields[2:]:
            fid, value = f.split(':')
            vector[int(fid)] = float(value)

        docno = comment.strip()
        if docno.startswith('docno:'):
            docno = docno[6:]  # docno:XXXX
        yield {'qid': qid, 'rel': rel, 'vector': vector, 'docno': docno}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', type=str)
    parser.add_argument('vector_file')
    parser.add_argument('output')
    args = parser.parse_args()

    fids_to_exclude = set()
    if args.exclude:
        for spec in args.exclude.split(','):
            if '-' not in spec:
                fids_to_exclude.add(int(spec))
            else:
                l, u = map(int, spec.split('-'))
                fids_to_exclude.update(range(l, u + 1))

    rows = list(parse_svmlight_lines(smart_open(args.vector_file)))
    max_fid = max(max(row['vector'].keys()) for row in rows)

    m = []
    for row in rows:
        v = np.zeros(max_fid)
        for fid, value in row['vector'].items():
            if fid in fids_to_exclude:
                continue
            v[fid - 1] = value  # note the shift
        m.append(v)

    X = np.vstack(m)
    print(X.shape)
    np.save(args.output, X)
