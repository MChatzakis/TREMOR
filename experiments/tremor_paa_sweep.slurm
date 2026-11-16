#!/bin/bash
#SBATCH --job-name=tremor-paa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=experiments/logs/tremor-paa.%j.out
#SBATCH --error=experiments/logs/tremor-paa.%j.err
# PAA-segment sweep for the optimized TREMOR build, one case per window length
# (1000, 992, 496, 3000), with DMASS on the same query subset. Everything runs
# under numactl --interleave=all. SET=maule|seifr.
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
module load fftw/3.3.9
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128
BIN=${BIN:-bin_new/tremor_main}
OUT=experiments/evals/tremor_paa_${SLURM_JOB_ID}; mkdir -p $OUT; cp $BIN $OUT/tremor_main
EV=experiments/evals/scaling_20260924
MW=/path/to/data/maule/waveforms
SW=/path/to/data/seifr/waveforms
NUMA=(numactl --interleave=all)
tremor() { # label paa case-array mode-args...
  local label=$1 paa=$2; local -n c=$3; shift 3
  srun --mpi=pmi2 -n 1 -c 128 "${NUMA[@]}" $OUT/tremor_main --dataset ${c[0]} --dataset-size ${c[1]} --queries ${c[2]} \
    --query-count ${c[3]} --window-length ${c[4]} --replication-groups 0 --index-threads 128 --search-workers 128 \
    --leaf-size 2000 --paa-segments $paa --pq-threshold-divisor 16 --benchmark-output $OUT/$label.timings.csv "$@" > $OUT/$label.log 2>&1
  echo "$label exit=$? $(tail -n +2 $OUT/$label.timings.csv 2>/dev/null)"
}
dmass() { # label case-array mode-args...
  local label=$1; local -n c=$2; shift 2
  srun --mpi=pmi2 -n 1 -c 128 "${NUMA[@]}" bin/dmass_main --dataset ${c[0]} --dataset-size ${c[1]} --queries ${c[2]} \
    --queries-size ${c[3]} --window-length ${c[4]} --replication-groups 0 --threads 128 \
    --benchmark-output $OUT/$label.timings.csv "$@" > $OUT/$label.log 2>&1
  echo "$label exit=$? $(tail -n +2 $OUT/$label.timings.csv 2>/dev/null)"
}
if [[ ${SET:-maule} = maule ]]; then
  GA=($MW/ZE_GA1S__HHE.len87316288.bin 87316288 $EV/inputs/maule/ZE_GA1S__HHE.q50.bin 10 1000)
  GC=($MW/ZE_GC2S__HHN.len613697643.bin 613697643 $EV/inputs/maule/ZE_GC2S__HHN.q50.bin 5 992)
  QF=($MW/XS_QF05_00_HHE.len410750008.bin 410750008 $EV/inputs/maule/XS_QF05_00_HHE.q50.bin 10 496)
  for th in 0.8 0.95; do
    for paa in 8 10 20; do tremor GA.paa$paa.th$th $paa GA --mode threshold --threshold $th --threshold-result-slots 1 --merge-offset 250 --output $OUT/GA.paa$paa.th$th; done
    for paa in 8 16; do tremor GC.paa$paa.th$th $paa GC --mode threshold --threshold $th --threshold-result-slots 1 --merge-offset 248 --output $OUT/GC.paa$paa.th$th; done
    for paa in 8 16; do tremor QF.paa$paa.th$th $paa QF --mode threshold --threshold $th --threshold-result-slots 1 --merge-offset 124 --output $OUT/QF.paa$paa.th$th; done
  done
  dmass GC.dmass GC --mode threshold-search --threshold 0.8 --merge-offset 248 --output $OUT/GC.dmass
  dmass QF.dmass QF --mode threshold-search --threshold 0.8 --merge-offset 124 --output $OUT/QF.dmass
else
  EP=($SW/RD_EPF__BHZ.len1046593001.bin 1046593001 $EV/inputs/seifr/RD_EPF__BHZ.q50.bin 5 3000)
  for paa in 8 10 12 15; do tremor EP.paa$paa $paa EP --mode knn --k 10 --merge-offset 0 --output $OUT/EP.paa$paa.csv; done
fi
