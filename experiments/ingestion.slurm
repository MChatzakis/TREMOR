#!/bin/bash
#SBATCH --job-name=tremor-ingestion
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=experiments/logs/tremor-ingestion.%j.out
#SBATCH --error=experiments/logs/tremor-ingestion.%j.err
#
# Index ingestion: build TREMOR's index over growing prefixes (1 GB, 2 GB, ...
# of raw float32 data) of the largest channel of each dataset of the evaluation, on the nodes of
# this allocation (sbatch --nodes N, no replication). Each build answers one
# template at threshold 0.99, so the run is almost pure index construction.
# Output: $OUT (default experiments/evals/ingestion_20260930)/ingestion.n<N>.<jobid>.csv with
# one row per build (index time and index memory of the slowest node).
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128

N=$SLURM_JOB_NUM_NODES
OUT=${OUT:-experiments/evals/ingestion_20260930}
mkdir -p $OUT/runs
cp -n bin/tremor_main $OUT/tremor_main
CSV=$OUT/ingestion.n$N.$SLURM_JOB_ID.csv
echo "dataset,channel,nodes,rep,samples,gb,index_time,index_gb" > $CSV

DATA=${DATA:-/path/to/data}  # the datasets of the paper
# dataset channel waveform total_samples window paa queries
CASES=(
  "maule ZE_G15S__HHN $DATA/maule/waveforms/ZE_G15S__HHN.len1823755814.bin 1823755814 992 16 $DATA/maule/templates/ZE_G15S__HHN.q50.bin"
  "seifr RD_ROSF__BHZ $DATA/seifr/waveforms/RD_ROSF__BHZ.len1576800001.bin 1576800001 3000 8 $DATA/seifr/templates/RD_ROSF__BHZ.q50.bin"
)

for rep in 1 2; do
  for c in "${CASES[@]}"; do
    read -r dataset channel waveform total window paa queries <<< "$c"
    # Prefixes of 1, 2, 3, ... GB (250M float32 points per GB), then the whole channel.
    sizes=$(seq 250000000 250000000 $((total - 1)); echo $total)
    for samples in $sizes; do
      name=$dataset.n$N.rep$rep.$samples
      srun --mpi=pmi2 -n $N numactl --interleave=all $OUT/tremor_main --dataset $waveform --dataset-size $samples \
        --queries $queries --query-count 1 --window-length $window --mode threshold --threshold 0.99 \
        --threshold-result-slots 1 --merge-offset $((window / 4)) --replication-groups 0 --index-threads 128 \
        --search-workers 128 --leaf-size 2000 --paa-segments $paa --pq-threshold-divisor 16 --refinement adaptive \
        --output $OUT/runs/$name --benchmark-output $OUT/runs/$name.timings.csv > $OUT/runs/$name.log 2>&1
      # Slowest node's index time and largest per-node index memory.
      read -r index_time index_gb < <(awk -F, 'NR > 1 { if ($2 > t) t = $2; if ($4 > g) g = $4 } END { print t, g }' \
        $OUT/runs/$name.timings.csv)
      echo "$dataset,$channel,$N,$rep,$samples,$(awk "BEGIN { print $samples * 4 / 1e9 }"),$index_time,$index_gb" >> $CSV
      rm -f $OUT/runs/${name}_*.csv  # threshold hits are not needed
    done
  done
done
