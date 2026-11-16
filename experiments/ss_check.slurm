#!/bin/bash
#SBATCH --job-name=ss-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=experiments/logs/ss-check.%j.out
# Sequential scan (SS, --method sequential) vs. SSS and TREMOR on one node with the same inputs, and TREMOR of
# the new build vs. the evaluated v20 binary: query time, and whether the answers (template, position) agree.
# Exits with an error if any answers differ.
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
OUT=${OUT:-/path/to/scratch/bins/ss_check}; mkdir -p $OUT
IN=experiments/evals/final_sweep_n4_20260925/inputs
GC=(--dataset /path/to/data/maule/waveforms/ZE_GC2S__HHN.len613697643.bin --dataset-size 613697643
    --queries $IN/maule/ZE_GC2S__HHN.q50.bin --query-count 10 --window-length 992 --paa-segments 16 --merge-offset 248)
EPF=(--dataset /path/to/data/seifr/waveforms/RD_EPF__BHZ.len1046593001.bin --dataset-size 1046593001
     --queries $IN/seifr/RD_EPF__BHZ.q50.bin --query-count 5 --window-length 3000 --paa-segments 8 --merge-offset 750)
VARIANTS=("tremor_v20 bin/tremor_main.v20 --refinement adaptive" "tremor bin/tremor_main --refinement adaptive"
          "sss bin/tremor_main --method skip-sequential" "ss bin/tremor_main --method sequential")
status=0
run() {  # name mode data...
  local name=$1 mode=$2; shift 2
  local reference=""
  for variant in "${VARIANTS[@]}"; do
    read -r label exe extra <<< "$variant"
    local args=(--mode threshold --threshold ${mode#th} --threshold-result-slots 1 --output $OUT/$name.$label) result=$OUT/$name.${label}_0.csv
    [[ $mode = k* ]] && { args=(--mode knn --k ${mode#k} --output $OUT/$name.$label.csv); result=$OUT/$name.$label.csv; }
    srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all $exe "$@" "${args[@]}" $extra --replication-groups 0 \
      --index-threads 128 --search-workers 128 --leaf-size 2000 --pq-threshold-divisor 16 \
      --benchmark-output $OUT/$name.$label.timings.csv > $OUT/$name.$label.log 2>&1
    local digest=$(tail -n +2 $result 2>/dev/null | cut -d, -f1,3 | tr -d ' ' | sort | md5sum | cut -c1-12)
    [[ -z $reference ]] && reference=$digest
    local same=$([[ $digest = $reference ]] && echo same || echo DIFFERENT)
    [[ $same = same ]] || status=1
    echo "$name $label: build/query (s) $(tail -1 $OUT/$name.$label.timings.csv | cut -d, -f2,3), answers $same"
  done
}
run gc th0.8 "${GC[@]}"
run gc th0.99 "${GC[@]}"
run gc th0.999 "${GC[@]}"
run epf k1 "${EPF[@]}"
run epf k10 "${EPF[@]}"
echo "finished (status $status)"
exit $status
