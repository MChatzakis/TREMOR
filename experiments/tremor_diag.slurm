#!/bin/bash
#SBATCH --job-name=tremor-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=experiments/logs/tremor-diag.%j.out
#SBATCH --error=experiments/logs/tremor-diag.%j.err
# Per-query pruning counters from the -DTREMOR_STATS build (bin_diag/).
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
EV=experiments/evals/scaling_20260924
OUT=experiments/evals/tremor_diag_${SLURM_JOB_ID}; mkdir -p $OUT
MW=/path/to/data/maule/waveforms
SW=/path/to/data/seifr/waveforms
for paa in 8 20; do
  echo "=== GA1S threshold paa=$paa"
  srun --mpi=pmi2 -n 1 -c 128 bin_diag/tremor_main --dataset $MW/ZE_GA1S__HHE.len87316288.bin --dataset-size 87316288 \
    --queries $EV/inputs/maule/ZE_GA1S__HHE.q50.bin --query-count 10 --window-length 1000 --mode threshold \
    --threshold 0.8 --threshold-result-slots 1 --merge-offset 250 --replication-groups 0 --index-threads 128 \
    --search-workers 128 --leaf-size 2000 --paa-segments $paa --pq-threshold-divisor 16 \
    --output $OUT/GA1S.paa$paa --benchmark-output $OUT/GA1S.paa$paa.timings.csv | grep -E "STATS"
done
echo "=== EPF knn paa=8"
srun --mpi=pmi2 -n 1 -c 128 bin_diag/tremor_main --dataset $SW/RD_EPF__BHZ.len1046593001.bin --dataset-size 1046593001 \
  --queries $EV/inputs/seifr/RD_EPF__BHZ.q50.bin --query-count 3 --window-length 3000 --mode knn --k 10 --merge-offset 0 \
  --replication-groups 0 --index-threads 128 --search-workers 128 --leaf-size 2000 --paa-segments 8 --pq-threshold-divisor 16 \
  --output $OUT/EPF.paa8.csv --benchmark-output $OUT/EPF.paa8.timings.csv | grep -E "STATS"
