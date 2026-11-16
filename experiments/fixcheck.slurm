#!/bin/bash
#SBATCH --job-name=tremor-fixcheck
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=experiments/logs/tremor-fixcheck.%j.out
#SBATCH --error=experiments/logs/tremor-fixcheck.%j.err
#
# Why is TREMOR slower than the sequential scan without lower bounds (SS) on Maule? One channel of the evaluation
# subset (CHANNEL, DATASET = maule or seifr), 4 nodes, full replication, the channel's 50 templates, in one job:
#   default   TREMOR as evaluated (f = 0.01, b = 0.05)
#   b0        b = 0: the scan never checks the lower bounds (no skip-sequential scan)
#   b0nofft   b = 0 and no FFT fallback (TREMOR_NO_FFT): the actual scan time of every template that TREMOR scans,
#             to compare with the scan estimates that default and b0 log
#   b0f0001   b = 0 and f = 0.001
#   ss        the sequential scan without lower bounds
#   stats     b0nofft with the diagnostic build (-DTREMOR_STATS): per-template setup / workers / post times, and
#             (second build) the per-worker scan times (SCANSTATS)
#   statsnopin stats with unpinned query workers (TREMOR_NO_PIN)
#   omp       b0nofft, but the scan runs with OpenMP on the main thread, as in SS (TREMOR_SCAN_OMP; threshold search)
#   ompstats  omp with the diagnostic build; ssstats: ss with the diagnostic build (blocks visited per template)
#   new       the binary's own defaults (no --scan-fallback-fraction, no TREMOR_SCAN_LB_FRACTION)
#   old       TREMOR as evaluated before 10-01 (OLD_BIN, f = 0.01, b = 0.05); with the fixed binary, default is f = 0.001, b = 0
#   b0nofftsleep, b0sleep: b0nofft and b0 with a 50 us pause between the polls of the scheduler thread (TREMOR_SCHED_SLEEP)
#   huge      b0nofft with the waveform and statistics in 2 MB aligned, MADV_HUGEPAGE memory (TREMOR_HUGE); hugestats: with the diagnostic build (PAGESTATS)
# CONFIGS selects the configurations, STATS_BIN the diagnostic build.
# Every run logs its per-template times and decisions; nothing is aggregated here (see the notebook).
# Output: $OUT/runs/<channel>.<setting>.<config>.{log,timings.csv,digest} and $OUT/fixcheck.<channel>.<jobid>.csv.
# Usage: sbatch --export=ALL,CHANNEL=ZE_G50S__HHN,DATASET=maule experiments/fixcheck.slurm
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_NUM_THREADS=128

OUT=${OUT:-experiments/evals/fixcheck_20261001}
BIN=$OUT/tremor_main              # copies, so that rebuilding bin/ does not change these runs
STATS_BIN=${STATS_BIN:-$OUT/tremor_main_stats}
CHANNEL=${CHANNEL:?set CHANNEL}
DATASET=${DATASET:-maule}
PLAN=experiments/evals/sub_tremor_main_20260929/plans/${DATASET}_threshold.tremor.n4.part001.tsv
read -r SAMPLES WAVEFORM QUERIES COUNT WINDOW OFFSET < <(awk -F'\t' -v c=$CHANNEL '$9 == c { print $15, $17, $18, $10, $11, $12; exit }' $PLAN)
PAA=16; [[ $WINDOW = 1000 ]] && PAA=10; [[ $WINDOW = 3000 ]] && PAA=8   # as in the evaluation
SETTINGS=${SETTINGS:-t0.8 t0.9 t0.99 t0.999 k1 k5}
[[ $DATASET = seifr ]] && SETTINGS=${SEIFR_SETTINGS:-t0.8 t0.9 k1 k5}
mkdir -p $OUT/runs
CSV=$OUT/fixcheck.$CHANNEL.$SLURM_JOB_ID.csv
echo "channel,setting,config,queries,query_time,index_time,same_answers" > $CSV

run() {  # setting config
  local setting=$1 config=$2 name=$CHANNEL.$1.$2 bin=$BIN
  local mode=(--mode threshold --threshold ${setting#t} --threshold-result-slots 1 --output $OUT/runs/$name)
  [[ $setting = k* ]] && mode=(--mode knn --k ${setting#k} --output $OUT/runs/$name.csv)
  local search=(--refinement adaptive --scan-fallback-fraction 0.01) env=(TREMOR_SCAN_LB_FRACTION=0)
  case $config in
    default) env=() ;;
    b0nofft) env+=(TREMOR_NO_FFT=1) ;;
    b0f0001) search=(--refinement adaptive --scan-fallback-fraction 0.001) ;;
    ss)      search=(--method sequential) env=() ;;
    stats)   env+=(TREMOR_NO_FFT=1); bin=$STATS_BIN ;;
    statsnopin) env+=(TREMOR_NO_FFT=1 TREMOR_NO_PIN=1); bin=$STATS_BIN ;;
    omp)     env+=(TREMOR_NO_FFT=1 TREMOR_SCAN_OMP=1) ;;
    ompstats) env+=(TREMOR_NO_FFT=1 TREMOR_SCAN_OMP=1); bin=$STATS_BIN ;;
    ssstats) search=(--method sequential) env=(); bin=$STATS_BIN ;;
    huge)    env+=(TREMOR_NO_FFT=1 TREMOR_HUGE=1) ;;
    b0nofftsleep) env+=(TREMOR_NO_FFT=1 TREMOR_SCHED_SLEEP=50) ;;
    b0sleep) env+=(TREMOR_SCHED_SLEEP=50) ;;
    old)     search=(--refinement adaptive --scan-fallback-fraction 0.01) env=(); bin=${OLD_BIN:?set OLD_BIN} ;;
    new)     search=(--refinement adaptive) env=() ;;  # the binary's own defaults
    hugestats) env+=(TREMOR_NO_FFT=1 TREMOR_HUGE=1); bin=$STATS_BIN ;;
  esac
  env "${env[@]}" srun --kill-on-bad-exit=1 --mpi=pmi2 -n $SLURM_NTASKS numactl --interleave=all $bin \
    --dataset $WAVEFORM --dataset-size $SAMPLES --queries $QUERIES --query-count $COUNT --window-length $WINDOW \
    --merge-offset $OFFSET --replication-groups 1 --index-threads 128 --search-workers 128 --leaf-size 2000 \
    --paa-segments $PAA --pq-threshold-divisor 16 "${mode[@]}" "${search[@]}" \
    --benchmark-output $OUT/runs/$name.timings.csv > $OUT/runs/$name.log 2>&1
  # answers: (template, position) of every detection or neighbor, over all nodes
  if [[ $setting = k* ]]; then tail -n +2 $OUT/runs/$name.csv | cut -d, -f1,3; else tail -q -n +2 $OUT/runs/${name}_*.csv | cut -d, -f1,3; fi \
    | sort | md5sum | cut -c1-12 > $OUT/runs/$name.digest
  rm -f $OUT/runs/${name}_*.csv
  local times=$(awk -F, 'NR > 1 { if ($2 > i) i = $2; if ($3 > q) q = $3 } END { print q "," i }' $OUT/runs/$name.timings.csv)
  local reference=$OUT/runs/$CHANNEL.$setting.default.digest
  [[ -f $reference ]] || reference=$OUT/runs/$CHANNEL.$setting.ss.digest
  echo "$CHANNEL,$setting,$config,$COUNT,$times,$(cmp -s $OUT/runs/$name.digest $reference && echo 1 || echo 0)" >> $CSV
}

# Warm-up (page cache of the waveform): one run whose result is not used.
(CSV=/dev/null; run ${SETTINGS%% *} default); rm -f $OUT/runs/$CHANNEL.${SETTINGS%% *}.default.*
for setting in $SETTINGS; do
  for config in ${CONFIGS:-default b0 b0nofft b0f0001 ss stats}; do run $setting $config; done
done
exit 0
