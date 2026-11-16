#!/bin/bash
#SBATCH --job-name=tremor-breakdown
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=experiments/logs/tremor-breakdown.%j.out
#SBATCH --error=experiments/logs/tremor-breakdown.%j.err
#
# Optimization breakdown: the same templates answered on one node by each
# optimization step of TREMOR, and by the skip-sequential scan (SSS):
#   s0 baseline      the TREMOR of the first evaluation (09-25)
#   s1 workers       + persistent query workers pinned to cores, reused buffers
#   s2 twopass       + two-pass leaf refinement (lower bounds, then distances)
#   s3 knnseed       + kNN: first bound from the query's own leaf, 2k-1 bound set
#   s3b skipbounds   + lower bounds of single windows skipped when the sampled
#                      bounds show they cannot prune (TREMOR-simple)
#   s4 fallback      s3 + fallback to the skip-sequential scan (first TREMOR-adaptive)
#   s5 adaptive      s4 + the scan skips useless lower bounds
#   s6 fft           s5 + FFT fallback when early abandoning cannot pay off (TREMOR)
# ONLY selects steps by label (regular expression); THRESHOLDS, MAULE_K and SEIFR_K select the settings.
# Workloads: Maule ZE.GC2S (613M points, 992-point templates) thresholds and k,
# and SeiFR RD.EPF (1.05G points, 3,000-point templates) k. Output:
# $OUT/breakdown.<jobid>.csv, one row per run, and one log per run with the
# per-template times; outputs are compared across steps (column "same_answers").
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128

OUT=${OUT:-experiments/evals/breakdown_20260927}
BINS=/path/to/scratch/breakdown_bins
mkdir -p $OUT/runs
CSV=$OUT/breakdown.$SLURM_JOB_ID.csv
echo "dataset,param,value,step,queries,query_time,index_time,same_answers" > $CSV

IN=experiments/evals/final_sweep_n4_20260925/inputs
GC=(--dataset /path/to/data/maule/waveforms/ZE_GC2S__HHN.len613697643.bin --dataset-size 613697643
    --queries $IN/maule/ZE_GC2S__HHN.q50.bin --query-count 10 --window-length 992 --paa-segments 16 --merge-offset 248)
EPF=(--dataset /path/to/data/seifr/waveforms/RD_EPF__BHZ.len1046593001.bin --dataset-size 1046593001
     --queries $IN/seifr/RD_EPF__BHZ.q50.bin --query-count 5 --window-length 3000 --paa-segments 8 --merge-offset 750)
STEPS=("s0_baseline s0_baseline" "s1_workers s1_workers" "s2_twopass s2_twopass" "s3_knnseed s3_knnseed"
       "s3b_skipbounds s5_skipbounds" "s4_fallback s4_fallback --refinement adaptive"
       "s5_adaptive s5_skipbounds --refinement adaptive" "s6_fft s6_fft --refinement adaptive"
       "sss s5_skipbounds --method skip-sequential")

run() {  # dataset param value data-args...
  local dataset=$1 param=$2 value=$3; shift 3
  local reference=""
  for step in "${STEPS[@]}"; do
    read -r label bin extra <<< "$step"
    [[ -z ${ONLY:-} || $label =~ $ONLY ]] || continue
    local name=$dataset.$param$value.$label
    local mode=(--mode threshold --threshold $value --threshold-result-slots 1 --output $OUT/runs/$name)
    local result=$OUT/runs/${name}_0.csv
    [[ $param = k ]] && { mode=(--mode knn --k $value --output $OUT/runs/$name.csv); result=$OUT/runs/$name.csv; }
    srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all $BINS/$bin "$@" "${mode[@]}" $extra --replication-groups 0 \
      --index-threads 128 --search-workers 128 --leaf-size 2000 --pq-threshold-divisor 16 \
      --benchmark-output $OUT/runs/$name.timings.csv > $OUT/runs/$name.log 2>&1
    local digest=$(tail -n +2 $result 2>/dev/null | cut -d, -f1,3 | sort | md5sum | cut -c1-12)
    [[ -z $reference ]] && reference=$digest
    local queries=$(grep -m1 -o "query-count [0-9]*" <<< "$*" | cut -d' ' -f2)
    local times=$(tail -1 $OUT/runs/$name.timings.csv)
    echo "$dataset,$param,$value,$label,$queries,$(cut -d, -f3 <<< "$times"),$(cut -d, -f2 <<< "$times"),$([[ $digest = $reference ]] && echo 1 || echo 0)" >> $CSV
    rm -f $OUT/runs/${name}_*.csv
  done
}

PARTS=${PARTS:-threshold knn}  # which workloads to run
if [[ $PARTS == *threshold* ]]; then
  for th in ${THRESHOLDS:-0.6 0.7 0.8 0.9 0.95 0.99 0.999}; do run maule th $th "${GC[@]}"; done
fi
if [[ $PARTS == *knn* ]]; then
  for k in ${MAULE_K-1 2 5 10}; do run maule k $k "${GC[@]}"; done
  for k in ${SEIFR_K-1 2 5 10}; do run seifr k $k "${EPF[@]}"; done
fi
