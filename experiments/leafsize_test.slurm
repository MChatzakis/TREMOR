#!/bin/bash
#SBATCH --job-name=tremor-leafsize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=experiments/logs/tremor-leafsize.%j.out
#
# One node, same allocation: leaf size of TREMOR's index (500 / 1000 / 2000 /
# 5000 subsequences) for the index-heavy workloads of Maule ZE.GC2S (threshold
# 0.99 and 0.999, k = 1), TREMOR-simple and TREMOR-adaptive, with SSS as reference.
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
BIN=${BIN:-/path/to/scratch/bins/lbskip}
OUT=/path/to/scratch/bins/leafsize_runs; mkdir -p $OUT
IN=experiments/evals/final_sweep_n4_20260925/inputs
GC=(--dataset /path/to/data/maule/waveforms/ZE_GC2S__HHN.len613697643.bin --dataset-size 613697643
    --queries $IN/maule/ZE_GC2S__HHN.q50.bin --query-count 20 --window-length 992 --paa-segments 16 --merge-offset 248)
COMMON=(--replication-groups 0 --index-threads 128 --search-workers 128 --pq-threshold-divisor 16)

run() {  # label method leaf workload-args...
  local label=$1 method=$2 leaf=$3; shift 3
  local mode=(--refinement $method); [[ $method = sss ]] && mode=(--method skip-sequential)
  srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all $BIN "${COMMON[@]}" --leaf-size $leaf "$@" "${mode[@]}" \
    --output $OUT/$label.$method.$leaf --benchmark-output $OUT/t.csv > $OUT/$label.$method.$leaf.log 2>&1
  echo "$label $method leaf=$leaf index_s=$(tail -1 $OUT/t.csv | cut -d, -f2) query_s=$(tail -1 $OUT/t.csv | cut -d, -f3) index_answered=$(grep -c '(index)' $OUT/$label.$method.$leaf.log)"
}

for w in "th0.99 --mode threshold --threshold 0.99 --threshold-result-slots 1" "th0.999 --mode threshold --threshold 0.999 --threshold-result-slots 1" "k1 --mode knn --k 1"; do
  set -- $w; label=$1; shift
  run $label sss 2000 "${GC[@]}" "$@"
  for leaf in 500 1000 2000 5000; do
    run $label adaptive $leaf "${GC[@]}" "$@"
    run $label leaf $leaf "${GC[@]}" "$@"
  done
done
