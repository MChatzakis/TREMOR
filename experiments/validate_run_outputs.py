#!/usr/bin/env python3
"""Reject malformed/partial timing and result CSVs before accepting a run."""
import csv
import math
from pathlib import Path
import sys


def rows(path):
    with Path(path).open() as f:
        yield from csv.DictReader(f, skipinitialspace=True)


def validate(prefix, workload, nodes, query_count, k):
    timings = list(rows(str(prefix) + '.timings.csv'))
    if sorted(int(r['node_rank']) for r in timings) != list(range(nodes)):
        raise ValueError('timing ranks must appear exactly once')
    for r in timings:
        for field in ['index_time', 'query_time']:
            if not math.isfinite(float(r[field])) or float(r[field]) < 0:
                raise ValueError('invalid timing value')
    if workload == 'knn':
        counts = [0] * query_count
        positions = [set() for _ in range(query_count)]
        for r in rows(str(prefix) + '.csv'):
            qi = int(r['tID'])
            rank = int(r['k'])
            pos = int(r['pos'])
            corr = float(r['corr'])
            if not (0 <= qi < query_count and 1 <= rank <= k and pos >= 0
                    and math.isfinite(corr) and -1.0002 <= corr <= 1.0002):
                raise ValueError('invalid kNN row')
            counts[qi] += 1
            if rank != counts[qi] or pos in positions[qi]:
                raise ValueError('duplicate/unordered kNN rank or position')
            positions[qi].add(pos)
        if counts != [k] * query_count:
            raise ValueError('missing kNN rows')
    else:
        for rank in range(nodes):
            with Path(str(prefix) + f'_{rank}.csv').open() as f:
                if f.readline().strip() != 'Query, Counter, Position, Cross Correlation':
                    raise ValueError('invalid threshold header')


if __name__ == '__main__':
    prefix, workload, nodes, query_count, k = sys.argv[1:]
    validate(prefix, workload, int(nodes), int(query_count), int(k) if k != 'NA' else 0)
