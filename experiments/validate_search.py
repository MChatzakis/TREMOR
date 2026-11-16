#!/usr/bin/env python3
"""Run both MPI executables against an independent Pearson-correlation oracle."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def read_rows(path):
    with path.open() as f:
        aliases = {'tID': 'Query', 'pos': 'Position', 'corr': 'Cross Correlation'}
        return [{aliases.get(k.strip(), k.strip()): v for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)]


def knn_with_exclusion(correlations, k, merge):
    """Canonical kNN with an exclusion zone: best first (ties by position),
    skipping windows within `merge` positions of one already kept."""
    order = sorted(range(len(correlations)), key=lambda p: (-correlations[p], p))
    kept = []
    for p in order:
        if len(kept) == k or correlations[p] == -np.inf:
            break
        if all(abs(p - q) >= merge for q in kept):
            kept.append(p)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['synthetic', 'maule', 'seifr'], default='synthetic')
    parser.add_argument('--bin-dir', type=Path, help='directory with tremor_main and dmass_main (default: <repo>/bin)')
    parser.add_argument('--paa-segments', type=int, default=8, help='TREMOR PAA segments (must divide the window)')
    parser.add_argument('--window', type=int, default=64, help='window length for the synthetic dataset')
    parser.add_argument('--refinement', choices=['leaf', 'adaptive'], default='leaf', help='TREMOR --refinement')
    parser.add_argument('--k', type=int, default=5, help='k for the kNN checks')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 2], help='MPI rank counts to test')
    parser.add_argument('--replication-groups', type=int, default=0, help='replication groups (0: one per rank)')
    parser.add_argument('--tremor-arg', action='append', default=[], help='extra TREMOR argument (repeatable), e.g. --tremor-arg=--scan-fallback-fraction=0')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bin_dir = (args.bin_dir or root / 'bin').resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(734)
    n, m, k = 8192, args.window, args.k
    x = rng.normal(size=n).astype('float32')
    x[2000:2128] = 0  # Constant windows must not become correlation=0.5 hits.
    # A near-silent stretch after loud data (like a gap or taper edge in the
    # real waveforms): prefix-sum statistics lose all precision here.
    x[5000:6000] *= 1e6
    x[6000:6600] *= 1e-3
    source = None
    if args.dataset != 'synthetic':
        metadata = root / 'dataprep' / ('seiFR' if args.dataset == 'seifr' else 'maule') / 'metadata'
        records = [json.loads(p.read_text()) for p in metadata.glob('*.json')]
        records = [r for r in records if r['kind'] == 'waveform' and r.get('pipeline_version') == 2
                   and r['number_of_points'] >= n and r['channel'].endswith('Z')]
        if not records:
            raise RuntimeError(f'No regenerated waveforms available for {args.dataset}')
        source = sorted(records, key=lambda r: r['binary_file_name'])[0]
        rate = source['frequency']['sampling_rate_hz']
        m = int(rate * (120 if args.dataset == 'seifr' else 10)) // 8 * 8
        # An interior slice avoids the daily taper's near-zero endpoint.
        offset = min(source['number_of_points'] - n, int(rate * 3600))
        with Path(source['binary_file']).open('rb') as f:
            f.seek(offset * 4)
            x = np.fromfile(f, dtype='float32', count=n)
    # Include the first/last subsequence and windows crossing MPI boundaries.
    positions = [0, 4090, min(6200, n - m - 1), n - m]
    queries = np.stack([x[p:p+m] for p in positions] + [rng.normal(size=m)])
    queries = queries.astype('float32')
    x.tofile(out / 'waveform.bin')
    queries.tofile(out / 'queries.bin')
    windows = np.lib.stride_tricks.sliding_window_view(x.astype('float64'), m)
    with np.errstate(invalid='ignore', divide='ignore'):
        zn = (windows - windows.mean(axis=1)[:, None]) / windows.std(axis=1)[:, None]
    qn = (queries.astype('float64') - queries.mean(axis=1, dtype='float64')[:, None])
    qn /= qn.std(axis=1)[:, None]
    correlations = qn @ zn.T / m
    correlations[:, windows.std(axis=1) <= 1e-12] = -np.inf
    checks = []
    failures = []
    for ranks in args.ranks:
        for merge in [0, m // 4]:
            for mode in ['knn', 'threshold']:
                for alg in ['dmass', 'tremor']:
                    prefix = out / f'{alg}.{mode}.n{ranks}.merge{merge}'
                    output = str(prefix) + ('.csv' if mode == 'knn' else '')
                    command = ['srun', '--mpi=pmi2', '--exclusive', '-N', str(ranks),
                               '-n', str(ranks), '--ntasks-per-node=1', '-c', '4',
                               str(bin_dir / f'{alg}_main'),
                               '--dataset', str(out / 'waveform.bin'), '--dataset-size', str(n),
                               '--queries', str(out / 'queries.bin'), '--window-length', str(m),
                               '--replication-groups', str(args.replication_groups), '--merge-offset', str(merge),
                               '--output', output, '--benchmark-output', str(prefix) + '.timings.csv']
                    if alg == 'dmass':
                        command += ['--queries-size', str(len(queries)), '--threads', '4',
                                    '--mode', mode + '-search']
                    else:
                        command += ['--query-count', str(len(queries)), '--index-threads', '4',
                                    '--search-workers', '4', '--paa-segments', str(args.paa_segments), '--refinement', args.refinement,
                                    *[part for a in args.tremor_arg for part in a.split('=', 1)],
                                    '--leaf-size', '64', '--mode', mode]
                    command += ['--k', str(k)] if mode == 'knn' else ['--threshold', '0.4']
                    with Path(str(prefix) + '.log').open('w') as log:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    if result.returncode:
                        failures.append(f'{prefix.name}: exit {result.returncode}')
                        continue
                    rows = read_rows(Path(output)) if mode == 'knn' else [
                        row for path in sorted(out.glob(prefix.name + '_*.csv')) for row in read_rows(path)]
                    for qi in range(len(queries)):
                        # Unfilled kNN slots (fewer than k answers) are written with an
                        # infinite distance, i.e. a correlation far below -1.
                        found = [r for r in rows if int(r['Query']) == qi and float(r['Cross Correlation']) >= -1.5]
                        actual = [int(r['Position']) for r in found]
                        cc = [float(r['Cross Correlation']) for r in found]
                        if any(p < 0 or p > n-m for p in actual):
                            failures.append(f'{prefix.name}: invalid positions query {qi}')
                            continue
                        if not np.allclose(cc, correlations[qi, actual], atol=2e-4):
                            failures.append(f'{prefix.name}: correlation mismatch query {qi}')
                        if mode == 'knn' and merge == 0:
                            expected = np.argsort(-correlations[qi])[:k].tolist()
                            if actual != expected:
                                failures.append(f'{prefix.name}: query {qi}: {actual} != {expected}')
                        elif mode == 'knn':
                            expected = knn_with_exclusion(correlations[qi], k, merge)
                            if actual != expected:
                                failures.append(f'{prefix.name}: query {qi}: {actual} != {expected}')
                        else:
                            expected = set()
                            parts = args.replication_groups or ranks  # data partitions
                            for rank in range(parts):
                                start = rank * (n // parts)
                                end = min((rank + 1) * (n // parts) if rank < parts-1 else n, n-m+1)
                                next_allowed = start
                                for pos in np.flatnonzero(correlations[qi, start:end] >= .4) + start:
                                    if pos >= next_allowed:
                                        expected.add(int(pos))
                                        next_allowed = pos + merge + 1
                            if set(actual) != expected or len(actual) != len(expected):
                                failures.append(f'{prefix.name}: threshold membership mismatch query {qi}')
                    timings = read_rows(Path(str(prefix) + '.timings.csv'))
                    if sorted(int(r['node_rank']) for r in timings) != list(range(ranks)):
                        failures.append(f'{prefix.name}: missing or duplicated timing ranks')
                    checks.append(prefix.name)
    report = {'dataset': args.dataset, 'source': source, 'checks': checks, 'failures': failures}
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
