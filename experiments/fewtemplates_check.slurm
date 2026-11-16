#!/bin/bash
#SBATCH --job-name=tremor-fewtpl
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/tremor-fewtpl.%j.out
#SBATCH --error=experiments/logs/tremor-fewtpl.%j.err
#
# Validates the fix for channels with no more templates than workers (10-06): with full replication, TREMOR used to
# hang when the coordinator's initial burst handed out every template (ZE_GA2S__HHN/HHZ: 3 templates; 4 or 8 nodes).
#   - ZE_GA2S__HHN and __HHZ with the fixed binary (NEW_BIN): the answers must equal DMASS-V3's
#     (large_maule_dmassv3_main_20261005; the answers do not depend on the number of nodes);
#   - XS_QF17_00_HHE (50 templates; 4 nodes only): the evaluated binary (OLD_BIN) and the fixed one must give the same
#     answers (the fix is not active there).
# Output: $OUT/fewtpl.n<nodes>.<jobid>.csv (channel, threshold, binary, query time, digest, same answers as reference).
# Usage: sbatch --nodes=4 --ntasks=4 experiments/fewtemplates_check.slurm   (and --nodes=8 --ntasks=8)
#        or several ranks on one node (the hang is in the MPI protocol): sbatch --nodes=1 --ntasks=4 --ntasks-per-node=4
#        --cpus-per-task=32 --export=ALL,THREADS=32 experiments/fewtemplates_check.slurm (QF17 runs on 4 nodes only)
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128

OUT=${OUT:-experiments/evals/fewtpl_check_20261006}
OLD_BIN=${OLD_BIN:-$OUT/tremor_main_old}
NEW_BIN=${NEW_BIN:-$OUT/tremor_main_new}
PLAN=experiments/evals/large_maule_tremor_main_20261005/plans/maule_threshold.tremor.n4.part001.tsv
DMASS=experiments/evals/large_maule_dmassv3_main_20261005/results/maule_threshold/dmass
N=$SLURM_NTASKS
THREADS=${THREADS:-128}
export OMP_NUM_THREADS=$THREADS
mkdir -p $OUT/runs
CSV=$OUT/fewtpl.n$N.nodes$SLURM_JOB_NUM_NODES.$SLURM_JOB_ID.csv
echo "channel,threshold,binary,query_time,digest,same_as_reference" > $CSV

digest() {  # threshold shards (prefix_<rank>.csv) -> digest of the (template, position) pairs
  tail -q -n +2 "$1"_*.csv | cut -d, -f1,3 | tr -d ' ' | sort | md5sum | cut -c1-12
}

run() {  # channel threshold binary-name binary reference-digest
  local channel=$1 t=$2 name=$1.t$2.$3.n$N.nodes$SLURM_JOB_NUM_NODES bin=$4 reference=$5
  read -r SAMPLES WAVEFORM QUERIES COUNT WINDOW OFFSET < <(awk -F'\t' -v c=$channel '$9 == c { print $15, $17, $18, $10, $11, $12; exit }' $PLAN)
  local paa=16; [[ $WINDOW = 1000 ]] && paa=10
  rm -f $OUT/runs/${name}_*.csv
  timeout 600 srun --kill-on-bad-exit=1 --mpi=pmi2 -n $N numactl --interleave=all $bin --dataset $WAVEFORM \
    --dataset-size $SAMPLES --queries $QUERIES --query-count $COUNT --window-length $WINDOW --mode threshold \
    --threshold $t --threshold-result-slots 1 --replication-groups 1 --index-threads $THREADS --search-workers $THREADS \
    --leaf-size 2000 --paa-segments $paa --pq-threshold-divisor 16 --merge-offset $OFFSET --refinement adaptive \
    --output $OUT/runs/$name --benchmark-output $OUT/runs/$name.timings.csv > $OUT/runs/$name.log 2>&1
  local status=$? q=NA d=NA
  if (( status == 0 )); then
    q=$(awk -F, 'NR > 1 && $3 > q { q = $3 } END { print q }' $OUT/runs/$name.timings.csv)
    d=$(digest $OUT/runs/$name)
  else
    d="exit$status"
  fi
  echo "$channel,$t,$3,$q,$d,$([[ $d = $reference ]] && echo 1 || echo 0)" >> $CSV
  echo "$channel t=$t $3: exit $status, query $q s, digest $d (reference $reference)"
}

for channel in ZE_GA2S__HHN ZE_GA2S__HHZ; do
  for t in 0.6 0.9 0.999; do
    tt=${t/./p}
    reference=$(digest $DMASS/maule_threshold.$channel.w1000.q3.th$tt.n4.r1)
    run $channel $t new $NEW_BIN $reference
  done
done
if (( N == 4 && SLURM_JOB_NUM_NODES == 4 )); then
  for t in 0.9 0.999; do
    run XS_QF17_00_HHE $t old $OLD_BIN none
    run XS_QF17_00_HHE $t new $NEW_BIN $(awk -F, -v t=$t '$1 == "XS_QF17_00_HHE" && $2 == t && $3 == "old" { print $5 }' $CSV)
  done
fi
exit 0
