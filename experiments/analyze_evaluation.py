#!/usr/bin/env python3
"""Export accepted, matched timings and plots for one evaluation manifest."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
import full_eval_analysis as analysis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('eval_root', type=Path)
    args = p.parse_args()
    root = args.eval_root.resolve()
    tables = analysis.load_eval_tables(root)
    manifest = tables['manifest']
    timings, errors = analysis.load_timings(root, manifest)
    skips = analysis.load_skips(root, manifest)
    failures = analysis.load_failures(root, manifest)
    running = analysis.load_running(root, manifest)
    status = analysis.completion_status(manifest, timings, skips, failures, running)
    paired = analysis.paired_speedups(timings)
    out = root / 'reports'
    out.mkdir(exist_ok=True)
    for name, table in [('timings_long', timings), ('parse_errors', errors),
                        ('completion_status', status), ('paired_speedups', paired),
                        ('skips', skips), ('failures', failures), ('running', running)]:
        table.to_csv(out / f'{name}.csv', index=False)
    fig = analysis.plot_completion(status)
    # Plot helpers return (figure, axes).
    (fig[0] if isinstance(fig, tuple) else fig).savefig(out / 'completion.png', dpi=160, bbox_inches='tight')
    plt.close('all')
    for ds, workload in [('maule', 'threshold'), ('seifr', 'knn')]:
        fig = analysis.plot_speedup_heatmap(paired, dataset=ds, workload=workload)
        (fig[0] if isinstance(fig, tuple) else fig).savefig(out / f'{ds}_speedup.png', dpi=160, bbox_inches='tight')
        plt.close('all')
    print(f'Wrote {out}; {len(paired)} matched comparisons; {len(errors)} timing parse errors')


if __name__ == '__main__':
    main()
