#!/bin/bash
#SBATCH --job-name=dmass-v3-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=experiments/logs/dmass-v3-check.%j.out
# DMASS-V1 vs. DMASS-V3 on one node with the same inputs: query time, and whether the answers are identical.
# Exits with an error if any answers differ. V1 outputs are reused from $OUT when present.
cd /path/to/TREMOR
module load intel/21U2/suite fftw/3.3.9
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
OUT=${OUT:-/path/to/scratch/bins/dmass_v3_check}; mkdir -p $OUT
IN=experiments/evals/final_sweep_n4_20260925/inputs
GC=(--dataset /path/to/data/maule/waveforms/ZE_GC2S__HHN.len613697643.bin --dataset-size 613697643
    --queries $IN/maule/ZE_GC2S__HHN.q50.bin --queries-size 10 --window-length 992 --merge-offset 248)
EPF=(--dataset /path/to/data/seifr/waveforms/RD_EPF__BHZ.len1046593001.bin --dataset-size 1046593001
     --queries $IN/seifr/RD_EPF__BHZ.q50.bin --queries-size 5 --window-length 3000 --merge-offset 750)
status=0
run() {  # name data mode...
  local name=$1; shift
  for version in v1 v3; do
    local exe=bin/dmass_main; [[ $version = v3 ]] && exe=bin/dmass_v3_main
    [[ $version = v1 && -f $OUT/$name.v1.timings.csv ]] && { echo "$name v1: $(tail -1 $OUT/$name.v1.timings.csv) (earlier run)"; continue; }
    srun --mpi=pmi2 -n 1 -c 128 numactl --interleave=all $exe "$@" --replication-groups 0 --threads 128 \
      --output $OUT/$name.$version --benchmark-output $OUT/$name.$version.timings.csv > $OUT/$name.$version.log 2>&1
    echo "$name $version: $(tail -1 $OUT/$name.$version.timings.csv)"
  done
  for f in $OUT/$name.v1*; do
    [[ $f = *.log || $f = *timings.csv ]] && continue
    if cmp -s <(sort $f) <(sort ${f/.v1/.v3}); then echo "  $(basename $f): identical"
    else echo "  $(basename $f): DIFFER"; status=1; fi
  done
}
run gc.th0.8 "${GC[@]}" --mode threshold-search --threshold 0.8
run gc.th0.999 "${GC[@]}" --mode threshold-search --threshold 0.999
run epf.k1 "${EPF[@]}" --mode knn-search --k 1
run epf.k10 "${EPF[@]}" --mode knn-search --k 10
echo "finished (status $status)"
exit $status
