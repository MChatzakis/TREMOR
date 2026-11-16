#!/bin/bash
#SBATCH --job-name=threads-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/threads-ab.%j.out
#SBATCH --error=experiments/logs/threads-ab.%j.err
# Same inputs as smoke_128threads.slurm (Maule GB0S, first 100 queries,
# threshold 0.8) at 64 threads, to decide the thread count per node.
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite fftw/3.3.9
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
T=${T:-64}
export OMP_NUM_THREADS=$T
OUT="experiments/evals/threads_ab_${SLURM_JOB_ID}"
mkdir -p "$OUT"
W=/path/to/data/maule/waveforms/ZE_GB0S__HHE.len356327321.bin; N=356327321
Q=/path/to/data/maule/templates/tem1005.ZE_GB0S__HHE.len1000.fmin1.0.fmax10.0.bin
srun --mpi=pmi2 -n 1 -c 128 bin/tremor_main --dataset $W --dataset-size $N --queries $Q --query-count 100 \
    --window-length 1000 --mode threshold --threshold 0.8 --threshold-result-slots 1 --merge-offset 250 \
    --replication-groups 0 --index-threads $T --search-workers $T --leaf-size 2000 --paa-segments 8 \
    --pq-threshold-divisor 16 --output "$OUT/tremor.t$T" --benchmark-output "$OUT/tremor.t$T.timings.csv"
srun --mpi=pmi2 -n 1 -c 128 bin/dmass_main --dataset $W --dataset-size $N --queries $Q --queries-size 100 \
    --window-length 1000 --mode threshold-search --threshold 0.8 --merge-offset 250 \
    --replication-groups 0 --threads $T --output "$OUT/dmass.t$T" --benchmark-output "$OUT/dmass.t$T.timings.csv"
for f in "$OUT"/*.timings.csv; do echo "$f: $(tail -n +2 "$f")"; done
