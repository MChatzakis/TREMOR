#!/bin/bash
#SBATCH --job-name=tremor-ablation
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=experiments/logs/tremor-ablation.%j.out
#SBATCH --error=experiments/logs/tremor-ablation.%j.err
#
# Ablation of TREMOR's parameters on one Maule channel of the evaluation subset (CHANNEL, e.g. ZE_G50S__HHN), with
# the setting of the evaluation: 4 nodes, full replication, the channel's 50 templates. One parameter varies at a
# time; the others keep their defaults (f = 0.01, b = 0.05, leaf size 2000, 16 PAA segments):
#   f     scan fallback fraction (--scan-fallback-fraction; "never": index only)   t = 0.99, 0.999 and k = 1
#   b     lower-bound cut-off of the scan (TREMOR_SCAN_LB_FRACTION)                 t = 0.99, 0.999
#   leaf  leaf size (--leaf-size)                                                   t = 0.999 and k = 1
#   paa   PAA segments (--paa-segments)                                             t = 0.999 and k = 1
#   scan  (not in the default PARAMS) the same templates without the index: the skip-sequential scan (sss), and the
#         sequential scan without lower bounds (ss)
# Output: $OUT/ablation.<channel>.<jobid>.csv, one row per run; the answers of every run are compared with those of
# the default configuration of the same setting (column "same_answers"). Repetitions go to their own directory
# (OUT=experiments/evals/ablation_20260930_rep2, ...); the notebook takes the median over the repetitions.
# PARAMS selects the parameters to vary (default "f b leaf paa"), SETTINGS the workloads of f and scan, FS the values
# of f, LEAVES the leaf sizes, BIN the binary (default: the binary of the evaluation, experiments/submit_subsets.sh with
# TAG=20260929). DEF_F and DEF_B are the defaults of f and b, B_SETTINGS the workloads of b, PARAM_SETTINGS those of
# leaf and paa.
# Usage: sbatch --export=ALL,CHANNEL=ZE_G50S__HHN[,OUT=...] experiments/ablation.slurm
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128

OUT=${OUT:-experiments/evals/ablation_20260930}
BIN=${BIN:-experiments/evals/sub_tremor_main_20260929/bin/tremor_main}
PLAN=experiments/evals/sub_tremor_main_20260929/plans/maule_threshold.tremor.n4.part001.tsv
CHANNEL=${CHANNEL:?set CHANNEL}
# waveform, samples, templates, template count and length, merge offset: from the evaluation plan
read -r SAMPLES WAVEFORM QUERIES COUNT WINDOW OFFSET < <(awk -F'\t' -v c=$CHANNEL '$9 == c { print $15, $17, $18, $10, $11, $12; exit }' $PLAN)
mkdir -p $OUT/runs
CSV=$OUT/ablation.$CHANNEL.$SLURM_JOB_ID.csv
echo "channel,samples,param,value,setting,queries,query_time,index_time,index_gb,index,same_answers" > $CSV

DEF_F=${DEF_F:-0.01} DEF_B=${DEF_B:-0.05}
run() {  # param value setting [f] [b] [leaf] [paa]
  local param=$1 value=$2 setting=$3 f=${4:-$DEF_F} b=${5:-$DEF_B} leaf=${6:-2000} paa=${7:-16}
  local name=$CHANNEL.$setting.f$f.b$b.leaf$leaf.paa$paa
  if [[ ! -f $OUT/runs/$name.timings.csv ]]; then  # the default configuration is shared by the four parameters
    local mode=(--mode threshold --threshold ${setting#t} --threshold-result-slots 1 --output $OUT/runs/$name)
    [[ $setting = k* ]] && mode=(--mode knn --k ${setting#k} --output $OUT/runs/$name.csv)
    local refinement=(--refinement adaptive --scan-fallback-fraction $f)
    [[ $f = never ]] && refinement=(--refinement leaf)
    [[ $f = sss ]] && refinement=(--method skip-sequential)  # the scans, for the same templates
    [[ $f = ss ]] && refinement=(--method sequential)
    env TREMOR_SCAN_LB_FRACTION=$b srun --kill-on-bad-exit=1 --mpi=pmi2 -n $SLURM_NTASKS numactl --interleave=all $BIN \
      --dataset $WAVEFORM --dataset-size $SAMPLES --queries $QUERIES --query-count $COUNT --window-length $WINDOW \
      --merge-offset $OFFSET --replication-groups 1 --index-threads 128 --search-workers 128 --leaf-size $leaf \
      --paa-segments $paa --pq-threshold-divisor 16 "${mode[@]}" "${refinement[@]}" \
      --benchmark-output $OUT/runs/$name.timings.csv > $OUT/runs/$name.log 2>&1
    # answers: (template, position) of every detection or neighbor, over all nodes
    if [[ $setting = k* ]]; then tail -n +2 $OUT/runs/$name.csv | cut -d, -f1,3; else tail -q -n +2 $OUT/runs/${name}_*.csv | cut -d, -f1,3; fi \
      | sort | md5sum | cut -c1-12 > $OUT/runs/$name.digest
    rm -f $OUT/runs/${name}_*.csv
  fi
  local reference=$OUT/runs/$CHANNEL.$setting.f$DEF_F.b$DEF_B.leaf2000.paa16.digest
  local times=$(awk -F, 'NR > 1 { if ($2 > i) i = $2; if ($3 > q) q = $3; if ($4 > g) g = $4 } END { print q "," i "," g }' $OUT/runs/$name.timings.csv)
  local index=$(grep -c -- '-> leaf' $OUT/runs/$name.log)  # templates answered through the index (adaptive runs)
  echo "$CHANNEL,$SAMPLES,$param,$value,$setting,$COUNT,$times,$index,$(cmp -s $OUT/runs/$name.digest $reference && echo 1 || echo 0)" >> $CSV
}

PARAMS=${PARAMS:-f b leaf paa}  # the parameters to vary
SETTINGS=${SETTINGS:-t0.99 t0.999 k1}  # the workloads of f and scan, e.g. "t0.8 t0.9"
# Warm-up (page cache of the waveform): one default run whose result is not used.
(CSV=/dev/null; run warmup 0 ${SETTINGS%% *}); rm -f $OUT/runs/$CHANNEL.${SETTINGS%% *}.f$DEF_F.b$DEF_B.leaf2000.paa16.*

for setting in $SETTINGS; do
  run f $DEF_F $setting  # the default configuration first: the reference answers
  [[ " $PARAMS " == *" f "* ]] && for f in ${FS:-0 0.001 0.1 never}; do run f $f $setting $f; done
  [[ " $PARAMS " == *" scan "* ]] && for scan in sss ss; do run scan $scan $setting $scan; done
done
for setting in ${B_SETTINGS:-t0.99 t0.999}; do
  [[ " $PARAMS " == *" b "* ]] && for b in 0 0.01 0.05 0.2 0.5 1; do run b $b $setting $DEF_F $b; done
done
for setting in ${PARAM_SETTINGS:-t0.999 k1}; do
  [[ " $PARAMS " == *" leaf "* ]] && for leaf in ${LEAVES:-500 1000 2000 4000 8000}; do run leaf $leaf $setting $DEF_F $DEF_B $leaf; done
  [[ " $PARAMS " == *" paa "* ]] && for paa in 4 8 16; do run paa $paa $setting $DEF_F $DEF_B 2000 $paa; done
done
exit 0
