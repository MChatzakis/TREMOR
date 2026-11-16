#!/bin/bash
#SBATCH --job-name=tremor-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=experiments/logs/tremor-profile.%j.out
#SBATCH --error=experiments/logs/tremor-profile.%j.err
# Per-query phase times from the -DTREMOR_STATS build (bin_diag/): tree
# traversal, gather/sort, and per-worker leaf processing (max/min/mean).
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
BIN=${BIN:-bin_diag/tremor_main}
EV=experiments/evals/scaling_20260924
OUT=experiments/evals/tremor_profile_${SLURM_JOB_ID}; mkdir -p $OUT
MW=/path/to/data/maule/waveforms
SW=/path/to/data/seifr/waveforms
run() { # name dataset n queries nq w mode extra...
  local name=$1 ds=$2 n=$3 q=$4 nq=$5 w=$6 mode=$7; shift 7
  echo "=== $name"
  local t0=$SECONDS
  srun --mpi=pmi2 -n 1 -c 128 $BIN --dataset $ds --dataset-size $n --queries $q --query-count $nq --window-length $w \
    --mode $mode --replication-groups 0 --index-threads 128 --leaf-size 2000 --pq-threshold-divisor 16 \
    --output $OUT/$name --benchmark-output $OUT/$name.timings.csv "$@" | grep -E "STATS|PHASES"
  echo "$name wall=$((SECONDS-t0))s timings: $(tail -n +2 $OUT/$name.timings.csv)"
}
run GA1S.w128 $MW/ZE_GA1S__HHE.len87316288.bin 87316288 $EV/inputs/maule/ZE_GA1S__HHE.q50.bin 5 1000 threshold \
  --threshold 0.8 --threshold-result-slots 1 --merge-offset 250 --search-workers 128 --paa-segments 8
run GA1S.w32 $MW/ZE_GA1S__HHE.len87316288.bin 87316288 $EV/inputs/maule/ZE_GA1S__HHE.q50.bin 5 1000 threshold \
  --threshold 0.8 --threshold-result-slots 1 --merge-offset 250 --search-workers 32 --paa-segments 8
run GA1S.th95 $MW/ZE_GA1S__HHE.len87316288.bin 87316288 $EV/inputs/maule/ZE_GA1S__HHE.q50.bin 5 1000 threshold \
  --threshold 0.95 --threshold-result-slots 1 --merge-offset 250 --search-workers 128 --paa-segments 8
run GC2S.w128 $MW/ZE_GC2S__HHN.len613697643.bin 613697643 $EV/inputs/maule/ZE_GC2S__HHN.q50.bin 3 992 threshold \
  --threshold 0.8 --threshold-result-slots 1 --merge-offset 248 --search-workers 128 --paa-segments 8
run EPF.w128 $SW/RD_EPF__BHZ.len1046593001.bin 1046593001 $EV/inputs/seifr/RD_EPF__BHZ.q50.bin 2 3000 knn \
  --k 10 --merge-offset 0 --search-workers 128 --paa-segments 8
