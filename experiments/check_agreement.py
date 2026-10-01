"""Answer agreement between the methods of the subset evaluation (roots sub_<method>_<kind>_<TAG>).

For every run of TREMOR, compares the answers (template, position) with those of SSS, DMASS-V3 and DMASS-V1 for the
same run id. Threshold runs: the union of the per-rank shards. k-NN runs: the (template, rank) -> position table.
Differences are classified: `boundary` when every differing detection has a correlation within 1e-4 of the threshold,
`tie` when the differing neighbors have correlations within 1e-4 of each other.

Usage: python experiments/check_agreement.py [TAG] [kind ...]   (default: 20260929 main scale repl8)
"""
import csv
import glob
import sys
from collections import Counter
from pathlib import Path

EVALS = Path(__file__).parent / 'evals'
TAG = sys.argv[1] if len(sys.argv) > 1 else '20260929'
KINDS = sys.argv[2:] or ['main', 'scale', 'repl8']
OTHERS = {'sss': ('tremor', 'SSS'), 'dmassv3': ('dmass', 'DMASS-V3'), 'dmassv1': ('dmass', 'DMASS-V1')}


def plan_rows(root, algorithm):
    rows = {}
    for plan in glob.glob(str(EVALS / root / 'plans' / f'*.{algorithm}.*.tsv')):
        for row in csv.DictReader(open(plan), delimiter='\t'):
            rows[row['run_id']] = row
    return rows


def answers(row):
    """{(template, position or rank): correlation}; None if the run has no complete output."""
    prefix = row['output_prefix']
    if not Path(prefix + '.timings.csv').exists():
        return None
    result = {}
    if row['threshold'] != 'NA':
        shards = glob.glob(prefix + '_*.csv')
        if not shards:
            return None
        for shard in shards:
            for line in list(open(shard))[1:]:
                query, _, position, corr = [v.strip() for v in line.split(',')]
                result[(int(query), int(position))] = float(corr)
    else:
        for line in list(open(prefix + '.csv'))[1:]:
            query, rank, position, corr = [v.strip() for v in line.split(',')]
            result[(int(query), int(rank))] = (int(position), float(corr))
    return result


def compare(a, b, threshold):
    if a == b or (threshold is None and {k: v[0] for k, v in a.items()} == {k: v[0] for k, v in b.items()}):
        return 'same'
    if threshold is not None:
        if set(a) == set(b):
            return 'same'
        differing = [a.get(k, b.get(k)) for k in set(a) ^ set(b)]
        return 'boundary' if all(abs(c - threshold) < 1e-4 for c in differing) else 'DIFFERENT'
    differing = [k for k in set(a) | set(b) if a.get(k, (None,))[0] != b.get(k, (None,))[0]]
    return 'tie' if all(k in a and k in b and abs(a[k][1] - b[k][1]) < 1e-4 for k in differing) else 'DIFFERENT'


for kind in KINDS:
    tremor = plan_rows(f'sub_tremor_{kind}_{TAG}', 'tremor')
    for method, (algorithm, name) in OTHERS.items():
        other = plan_rows(f'sub_{method}_{kind}_{TAG}', algorithm)
        counts, examples = Counter(), []
        for run, row in tremor.items():
            a, b = answers(row), answers(other[run]) if run in other else None
            if a is None or b is None:
                counts['missing'] += 1
                continue
            verdict = compare(a, b, float(row['threshold']) if row['threshold'] != 'NA' else None)
            counts[verdict] += 1
            if verdict == 'DIFFERENT' and len(examples) < 3:
                examples.append(run)
        print(f'{kind}: TREMOR vs {name}: {dict(counts)}' + (f' e.g. {examples}' if examples else ''))
