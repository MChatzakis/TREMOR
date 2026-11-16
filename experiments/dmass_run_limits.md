# DMASS and TREMOR run limits

Computed from `/path/to/TREMOR/experiments/evals/large_seifr_tremor_none8_20261005/cases`. Guards come from single-node peak-memory probes (experiments/probe_limits.slurm).

One MPI rank runs per node. With replication groups = 0, each rank owns a contiguous partition plus up to window_length - 1 overlap samples.

- DMASS uses 32-bit real FFTs of the 7-smooth length >= local samples + window - 1, so local samples must stay below INT_MAX = 2,147,483,647. It caches the chunk spectrum once per rank. Measured peak RSS: 66 GB at 1.07G local samples.
- Measured TREMOR peak RSS (64 threads): 40 GB at 600M, 78 GB at 1.2G, 165 GB at 2.57G local samples (Maule), 81 GB at 1.58G (SeiFR). Nodes have 247 GB.
- The DMASS guard is 2,100,000,000 local samples; the TREMOR guard is 2,600,000,000. Unsupported rows receive explicit skip records.
- Slurm allocates exclusive nodes and all node memory. TREMOR index memory also depends on the signal and leaf splits.
- Compare only paired successful runs. A missing, skipped, failed, or timed-out run is not a speedup measurement.

| Dataset | Nodes | Cases | TREMOR eligible | DMASS eligible | DMASS FFT-safe | Maximum local samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| maule | 8 | 243 | 243 | 243 | 243 | 227,970,467 |
| seifr | 8 | 7 | 7 | 7 | 7 | 197,102,999 |

The default row timeout is 3 hours (per-node-count overrides: 1=2400 2=1800 4=1800 8=1800) and timed-out rows are not retried. Each result file is limited to 1 GiB while writing. The query sample size and seed are recorded in run_config.tsv; the original query indices are saved beside sampled input binaries.
