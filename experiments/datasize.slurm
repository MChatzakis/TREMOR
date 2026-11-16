#!/bin/bash
#SBATCH --job-name=tremor-datasize
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=experiments/logs/tremor-datasize.%j.out
#SBATCH --error=experiments/logs/tremor-datasize.%j.err
#
# Query answering time for growing prefixes (250M, 500M, ... points, then the whole channel) of the largest channel
# of a dataset, with the setting and the binary of the evaluation (sub_tremor_main_20260929): the nodes of the
# allocation (default 4), full replication, the channel's 50 templates; Maule: threshold search, t = 0.9; SeiFR: k-NN
# search, k = 5. DATASETS: the datasets, one after the other (default: DATASET).
# Output: $OUT/datasize.<dataset>.n<nodes>.rep<REP>.<jobid>.csv, one row per prefix (slowest node's index and query
# time); the answers of every run are kept in $OUT/runs.
# Usage: sbatch [--nodes=1 --ntasks=1] --export=ALL,DATASETS="maule seifr",REP=1 experiments/datasize.slurm
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128

OUT=${OUT:-experiments/evals/datasize_20261001}
EVAL=experiments/evals/sub_tremor_main_20260929
BIN=$EVAL/bin/tremor_main
N=$SLURM_JOB_NUM_NODES REP=${REP:-1}
for DATASET in ${DATASETS:-${DATASET:?set DATASETS}}; do
case $DATASET in
  maule) channel=ZE_G15S__HHN total=1823755814 window=992 paa=16
         waveform=/path/to/data/maule/waveforms/ZE_G15S__HHN.len1823755814.bin
         mode=(--mode threshold --threshold 0.9 --threshold-result-slots 1) ;;
  seifr) channel=RD_ROSF__BHZ total=1576800001 window=3000 paa=8
         waveform=/path/to/data/seifr/waveforms/RD_ROSF__BHZ.len1576800001.bin
         mode=(--mode knn --k 5) ;;
esac
queries=$EVAL/inputs/$DATASET/$channel.q50.bin
mkdir -p $OUT/runs
CSV=$OUT/datasize.$DATASET.n$N.rep$REP.$SLURM_JOB_ID.csv
echo "dataset,channel,nodes,rep,samples,index_time,query_time" > $CSV

for samples in 250000000 500000000 750000000 1000000000 1250000000 1500000000 $total; do
  out=$OUT/runs/$DATASET.n$N.rep$REP.$samples
  output=(--output $out)
  [[ $DATASET = seifr ]] && output=(--output $out.csv)
  srun --kill-on-bad-exit=1 --mpi=pmi2 -n $N numactl --interleave=all $BIN --dataset $waveform --dataset-size $samples \
    --queries $queries --query-count 50 --window-length $window "${mode[@]}" --replication-groups 1 \
    --index-threads 128 --search-workers 128 --leaf-size 2000 --paa-segments $paa --pq-threshold-divisor 16 \
    --merge-offset $((window / 4)) --refinement adaptive "${output[@]}" --benchmark-output $out.timings.csv \
    > $out.log 2>&1
  read -r index_time query_time < <(awk -F, 'NR > 1 { if ($2 > i) i = $2; if ($3 > q) q = $3 } END { print i, q }' \
    $out.timings.csv)
  echo "$DATASET,$channel,$N,$REP,$samples,$index_time,$query_time" >> $CSV
done
done
