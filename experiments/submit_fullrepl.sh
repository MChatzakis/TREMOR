#!/bin/bash
# Query-time comparison of D-MASS, SSS, TREMOR-simple and TREMOR-adaptive under
# full replication (every node holds the whole waveform; templates are
# scheduled dynamically). Uses the current bin/ binaries.
#   PHASE=main:  4 nodes, Maule thresholds 0.7-0.999, Maule k and SeiFR k in {1, 10, 100}
#                (roots full_{adaptive,sss,simple}_<TAG>; D-MASS runs in full_adaptive)
#   PHASE=scale: TREMOR on 1, 2, 8 nodes (full) and on 8 nodes with groups 0/2/4, SSS on
#                2 nodes (full), Maule th 0.8 and SeiFR k 10 (roots scale_*_<TAG>, repl8_*_<TAG>).
#                D-MASS and the other SSS configurations come from earlier roots (unchanged code).
# Usage: PHASE=main TAG=20260928 bash experiments/submit_fullrepl.sh
set -euo pipefail
cd /path/to/TREMOR
PHASE=${PHASE:?set PHASE=main or PHASE=scale}
TAG=${TAG:?set TAG}
ENVFILE=${ENVFILE:-$PWD/experiments/fullrepl_env.sh}
source $ENVFILE
export LAUNCH_WRAPPER="numactl --interleave=all" PAA_BY_WINDOW="1000=10 992=16 496=16 3000=8"
export ROW_TIMEOUT_BY_NODES="1=14400 2=10800 4=10800 8=5400"

submit() {  # root nodes groups keep_dmass env...
  local root=$1 nodes=$2 groups=$3 keep=$4; shift 4
  env "$@" NODE_COUNTS="$nodes" REPLICATION_GROUPS="$groups" EVAL_ROOT=$PWD/experiments/evals/$root \
    bash experiments/submit_full_evaluation.sh > experiments/logs/submit_$root.log 2>&1
  [[ $keep = 1 ]] || scancel $(awk -F'\t' 'NR>1 && $4=="dmass"{print $9}' experiments/evals/$root/submissions.tsv)
  echo "$root: $(awk -F'\t' -v keep=$keep 'NR>1 && ($4=="tremor" || keep==1){print $2"."$4":"$9"("$6")"}' experiments/evals/$root/submissions.tsv | tr '\n' ' ')"
}

if [[ $PHASE = main ]]; then
  GRID=(WORKLOADS="maule_threshold maule_knn seifr_knn" MAULE_THRESHOLDS="0.7 0.8 0.9 0.95 0.99 0.999"
        MAULE_K_VALUES="1 10 100" SEIFR_K_VALUES="1 10 100")
  submit full_adaptive_$TAG 4 1 1 "${GRID[@]}" TREMOR_REFINEMENT=adaptive
  submit full_sss_$TAG 4 1 0 "${GRID[@]}" TREMOR_METHOD=skip-sequential
  submit full_simple_$TAG 4 1 0 "${GRID[@]}" TREMOR_REFINEMENT=leaf
else
  SCALE=(WORKLOADS="maule_threshold seifr_knn" MAULE_THRESHOLDS=0.8 SEIFR_K_VALUES=10)
  # Shorter row timeouts (and thus array time limits) let the runs backfill.
  FAST="1=3600 2=2400 8=1800"; SLOW="1=9000 2=5400 8=3600"
  for method in adaptive simple; do
    refinement=$([[ $method = simple ]] && echo leaf || echo adaptive)
    timeouts=$([[ $method = simple ]] && echo "$SLOW" || echo "$FAST")
    submit scale_${method}_$TAG "1 2 8" 1 0 "${SCALE[@]}" TREMOR_REFINEMENT=$refinement ROW_TIMEOUT_BY_NODES="$timeouts"
    submit repl8_${method}_$TAG 8 "0 2 4" 0 "${SCALE[@]}" TREMOR_REFINEMENT=$refinement ROW_TIMEOUT_BY_NODES="$timeouts"
  done
  submit scale_sss_$TAG 2 1 0 "${SCALE[@]}" TREMOR_METHOD=skip-sequential ROW_TIMEOUT_BY_NODES="$FAST"
fi
