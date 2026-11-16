#!/bin/bash
#SBATCH --job-name=fft-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=experiments/logs/fft-check.%j.out
# TREMOR with (VARIANT=fft) or without (VARIANT=nofft, TREMOR_NO_FFT=1) the FFT fallback, one node, on SeiFR RD.EPF
# (10 templates) and Maule ZE.GC2S (10 templates). Writes answers and timings to $OUT/<config>.<variant>*; compare the
# two variants with experiments/fft_compare.sh.
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
VARIANT=${VARIANT:?fft or nofft}
[[ $VARIANT = nofft ]] && export TREMOR_NO_FFT=1
OUT=${OUT:-/path/to/scratch/bins/fft_check}; mkdir -p $OUT
IN=experiments/evals/final_sweep_n4_20260925/inputs
EPF=(--dataset /path/to/data/seifr/waveforms/RD_EPF__BHZ.len1046593001.bin --dataset-size 1046593001
     --queries $IN/seifr/RD_EPF__BHZ.q50.bin --query-count 10 --window-length 3000 --paa-segments 8 --merge-offset 750)
GC=(--dataset /path/to/data/maule/waveforms/ZE_GC2S__HHN.len613697643.bin --dataset-size 613697643
    --queries $IN/maule/ZE_GC2S__HHN.q50.bin --query-count 10 --window-length 992 --paa-segments 16 --merge-offset 248)
run() {  # name mode data...
  local name=$1 mode=$2; shift 2
  local args=(--mode threshold --threshold ${mode#th} --threshold-result-slots 1 --output $OUT/$name.$VARIANT)
  [[ $mode = k* ]] && args=(--mode knn --k ${mode#k} --output $OUT/$name.$VARIANT.csv)
  srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all bin/tremor_main "$@" "${args[@]}" --refinement adaptive \
    --replication-groups 0 --index-threads 128 --search-workers 128 --leaf-size 2000 --pq-threshold-divisor 16 \
    --benchmark-output $OUT/$name.$VARIANT.timings.csv > $OUT/$name.$VARIANT.log 2>&1
  echo "$name $VARIANT: build/query (s) $(tail -1 $OUT/$name.$VARIANT.timings.csv | cut -d, -f2,3);" \
       "templates by strategy: $(grep -o 's (\(index\|scan\|fft\))' $OUT/$name.$VARIANT.log | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
}
for mode in k2 k10 th0.6 th0.9; do run epf.$mode $mode "${EPF[@]}"; done
for mode in th0.6 th0.8 th0.999 k5; do run gc.$mode $mode "${GC[@]}"; done
echo finished
