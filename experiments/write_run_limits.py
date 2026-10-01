#!/usr/bin/env python3
"""Refresh memory/FFT eligibility notes from the actual evaluation cases."""
import argparse
import csv
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('eval_root', type=Path)
    args = p.parse_args()
    tremor_max = int(os.environ.get('TREMOR_MAX_LOCAL_SAMPLES', 600000000))
    dmass_max = int(os.environ.get('DMASS_MAX_LOCAL_SAMPLES', 350000000))
    timeout_h = int(os.environ.get('ROW_TIMEOUT_SECONDS', 10800)) / 3600
    lines = ['# DMASS and TREMOR run limits', '',
             f'Computed from `{args.eval_root}/cases`. Guards come from single-node peak-memory probes.', '',
             'One MPI rank runs per node. With replication groups = 0, each rank owns a contiguous partition plus up to window_length - 1 overlap samples.', '',
             '- DMASS uses 32-bit real FFTs of the 7-smooth length >= local samples + window - 1, so local samples must stay below INT_MAX = 2,147,483,647. It caches the chunk spectrum once per rank. Measured peak RSS: 66 GB at 1.07G local samples.',
             '- Measured TREMOR peak RSS (64 threads): 40 GB at 600M, 78 GB at 1.2G, 165 GB at 2.57G local samples (Maule), 81 GB at 1.58G (SeiFR). Nodes have 247 GB.',
             f'- The DMASS guard is {dmass_max:,} local samples; the TREMOR guard is {tremor_max:,}. Unsupported rows receive explicit skip records.',
             '- Slurm allocates exclusive nodes and all node memory. TREMOR index memory also depends on the signal and leaf splits.',
             '- Compare only paired successful runs. A missing, skipped, failed, or timed-out run is not a speedup measurement.', '',
             '| Dataset | Nodes | Cases | TREMOR eligible | DMASS eligible | DMASS FFT-safe | Maximum local samples |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for ds in ['maule','seifr']:
        with (args.eval_root / 'cases' / f'{ds}_cases.tsv').open() as f:
            cases = list(csv.DictReader(f, delimiter='\t'))
        for nodes in [int(x) for x in os.environ.get('NODE_COUNTS', '1 2 4 6 8 10').split()]:
            local = []
            for c in cases:
                n,m = int(c['waveform_samples']),int(c['window_length'])
                base = n//nodes
                local.append(max(base + (m-1 if nodes>1 else 0), n-base*(nodes-1)))
            lines.append(f'| {ds} | {nodes} | {len(cases)} | {sum(n<=tremor_max for n in local)} | {sum(n<=dmass_max for n in local)} | {sum(n+int(3000)<=2147483647 for n in local)} | {max(local,default=0):,} |')
    lines += ['', f'The default row timeout is {timeout_h:g} hours (per-node-count overrides: {os.environ.get("ROW_TIMEOUT_BY_NODES") or "none"}) and timed-out rows are not retried. Each result file is limited to 1 GiB while writing. The query sample size and seed are recorded in run_config.tsv; the original query indices are saved beside sampled input binaries.', '']
    (Path(__file__).parent / 'dmass_run_limits.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
