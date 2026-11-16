#!/bin/bash
#SBATCH --job-name=tremor-lbskip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=experiments/logs/tremor-lbskip.%j.out
#
# One node, same allocation: SSS vs TREMOR-adaptive with the scan's lower-bound
# check always on (lb1.1), or skipped when the sampled candidate fraction is at
# least 0.9 / 0.5. Prints total query time, and an answer digest per method.
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
BIN=${BIN:-/path/to/scratch/bins/lbskip}
OUT=/path/to/scratch/bins/lbskip_runs; mkdir -p $OUT
IN=experiments/evals/final_sweep_n4_20260925/inputs
GC=(--dataset /path/to/data/maule/waveforms/ZE_GC2S__HHN.len613697643.bin --dataset-size 613697643
    --queries $IN/maule/ZE_GC2S__HHN.q50.bin --query-count 10 --window-length 992 --paa-segments 16 --merge-offset 248)
EPF=(--dataset /path/to/data/seifr/waveforms/RD_EPF__BHZ.len1046593001.bin --dataset-size 1046593001
     --queries $IN/seifr/RD_EPF__BHZ.q50.bin --query-count 5 --window-length 3000 --paa-segments 8 --merge-offset 750)
COMMON=(--replication-groups 0 --index-threads 128 --search-workers 128 --leaf-size 2000 --pq-threshold-divisor 16)

run() {  # label workload-args...
  local label=$1; shift
  for m in sss lb1.1 lb0.9 lb0.5; do
    local mode=(--refinement adaptive); [[ $m = sss ]] && mode=(--method skip-sequential)
    TREMOR_SCAN_LB_FRACTION=${m#lb} srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all $BIN "${COMMON[@]}" "$@" "${mode[@]}" \
      --output $OUT/$label.$m --benchmark-output $OUT/$label.$m.timings.csv > $OUT/$label.$m.log 2>&1
    local result=$OUT/$label.${m}_0.csv; [[ -f $OUT/$label.$m.csv ]] && result=$OUT/$label.$m.csv
    echo "$label $m query_s=$(tail -1 $OUT/$label.$m.timings.csv | cut -d, -f3) answers=$(tail -n +2 $result | cut -d, -f1,3 | sort | md5sum | cut -c1-8)"
  done
}

for th in 0.7 0.8 0.9 0.95 0.99 0.999; do run GC.th$th "${GC[@]}" --mode threshold --threshold $th --threshold-result-slots 1; done
for k in 1 10; do run GC.k$k "${GC[@]}" --mode knn --k $k; done
for k in 1 10; do run EPF.k$k "${EPF[@]}" --mode knn --k $k; done
