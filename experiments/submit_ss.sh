#!/bin/bash
# Sequential scan (SS, --method sequential) on the subset used for the parameter sweeps (8 Maule channels, the
# 7 SeiFR channels, 50 templates each), in the configurations of the other methods:
#   ss_full_<TAG>:  4 nodes, full replication; Maule t = 0.7-0.999 and k = 1, 10, 100; SeiFR k = 1-100
#   ss_scale_<TAG>: 1, 2, 8 nodes, full replication; Maule t = 0.8, SeiFR k = 10
#   ss_repl8_<TAG>: 8 nodes, replication groups 0/2/4; Maule t = 0.8, SeiFR k = 10
# Roots that were already submitted are skipped.
# Usage: TAG=20260929 [DEPENDENCY=afterok:<job>] bash experiments/submit_ss.sh
set -euo pipefail
cd /path/to/TREMOR
TAG=${TAG:?set TAG}
source experiments/fullrepl_env.sh
export LAUNCH_WRAPPER="numactl --interleave=all" PAA_BY_WINDOW="1000=10 992=16 496=16 3000=8"
export ROW_TIMEOUT_BY_NODES="1=7200 2=3600 4=3600 8=1800" ALGORITHMS=tremor TREMOR_METHOD=sequential
export SUBMIT_DEPENDENCY=${DEPENDENCY:-}

submit() {  # root nodes groups env...
  local root=$1 nodes=$2 groups=$3; shift 3
  [[ -f experiments/evals/$root/submissions.tsv ]] && { echo "$root: already submitted"; return; }
  env "$@" NODE_COUNTS="$nodes" REPLICATION_GROUPS="$groups" EVAL_ROOT=$PWD/experiments/evals/$root \
    bash experiments/submit_full_evaluation.sh > experiments/logs/submit_$root.log 2>&1
  echo "$root: $(awk -F'\t' 'NR>1{print $2"."$3".n"$5":"$9"("$7" runs)"}' experiments/evals/$root/submissions.tsv | tr '\n' ' ')"
}

submit ss_full_$TAG 4 1 WORKLOADS="maule_threshold maule_knn seifr_knn" MAULE_THRESHOLDS="0.7 0.8 0.9 0.95 0.99 0.999" \
  MAULE_K_VALUES="1 10 100" SEIFR_K_VALUES="1 2 4 6 8 10 20 100"
submit ss_scale_$TAG "1 2 8" 1 WORKLOADS="maule_threshold seifr_knn" MAULE_THRESHOLDS=0.8 SEIFR_K_VALUES=10
submit ss_repl8_$TAG 8 "0 2 4" WORKLOADS="maule_threshold seifr_knn" MAULE_THRESHOLDS=0.8 SEIFR_K_VALUES=10
