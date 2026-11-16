#!/bin/bash
# DMASS-V3 (dmass/DMASS_V3.c, bin/dmass_v3_main) runs for the configurations of the full-replication
# evaluation; DMASS-V1 comes from the earlier roots. Only the DMASS runs are submitted.
#   dmassv3_full_<TAG>:  4 nodes, full replication, Maule thresholds 0.7-0.999 and SeiFR k in {1, 10, 100}
#   dmassv3_scale_<TAG>: 1, 2, 8 nodes, full replication, Maule t = 0.8 and SeiFR k = 10
#   dmassv3_repl8_<TAG>: 8 nodes, replication groups 0/2/4, Maule t = 0.8 and SeiFR k = 10
#   dmassv3_mauleknn_<TAG>: 4 nodes, full replication, Maule k in {1, 10, 100}
# Roots that were already submitted are skipped.
# Usage: TAG=20260929 [DEPENDENCY=afterok:<job>] bash experiments/submit_dmass_v3.sh
set -euo pipefail
cd /path/to/TREMOR
TAG=${TAG:?set TAG}
source experiments/fullrepl_env.sh
export LAUNCH_WRAPPER="numactl --interleave=all" ALGORITHMS=dmass MASS_BIN=$PWD/bin/dmass_v3_main
export ROW_TIMEOUT_BY_NODES="1=7200 2=3600 4=3600 8=1800" SUBMIT_DEPENDENCY=${DEPENDENCY:-}

submit() {  # root nodes groups env...
  local root=$1 nodes=$2 groups=$3; shift 3
  [[ -f experiments/evals/$root/submissions.tsv ]] && { echo "$root: already submitted"; return; }
  env "$@" NODE_COUNTS="$nodes" REPLICATION_GROUPS="$groups" EVAL_ROOT=$PWD/experiments/evals/$root \
    bash experiments/submit_full_evaluation.sh > experiments/logs/submit_$root.log 2>&1
  echo "$root: $(awk -F'\t' 'NR>1{print $2"."$3".n"$5":"$9}' experiments/evals/$root/submissions.tsv | tr '\n' ' ')"
}

submit dmassv3_full_$TAG 4 1 WORKLOADS="maule_threshold seifr_knn" MAULE_THRESHOLDS="0.7 0.8 0.9 0.95 0.99 0.999" SEIFR_K_VALUES="1 10 100"
submit dmassv3_scale_$TAG "1 2 8" 1 WORKLOADS="maule_threshold seifr_knn" MAULE_THRESHOLDS=0.8 SEIFR_K_VALUES=10
submit dmassv3_repl8_$TAG 8 "0 2 4" WORKLOADS="maule_threshold seifr_knn" MAULE_THRESHOLDS=0.8 SEIFR_K_VALUES=10
submit dmassv3_mauleknn_$TAG 4 1 WORKLOADS=maule_knn MAULE_K_VALUES="1 10 100"
