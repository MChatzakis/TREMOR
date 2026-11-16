#!/bin/bash
# SeiFR k-NN with small k: TREMOR, SSS, DMASS-V1 and DMASS-V3 on 4 nodes with full replication, with the setup
# of the full-replication evaluation (k = 1, 10, 100 come from full_*_20260928 and dmassv3_full_20260929).
#   seifrk_adaptive_<TAG>: TREMOR and DMASS-V1;  seifrk_sss_<TAG>: SSS;  seifrk_dmassv3_<TAG>: DMASS-V3
# Usage: TAG=20260929 [K_VALUES="2 4 6 8 20"] bash experiments/submit_seifr_k.sh
set -euo pipefail
cd /path/to/TREMOR
TAG=${TAG:?set TAG}
source experiments/fullrepl_env.sh
export LAUNCH_WRAPPER="numactl --interleave=all" PAA_BY_WINDOW="1000=10 992=16 496=16 3000=8"
export ROW_TIMEOUT_BY_NODES="4=10800" WORKLOADS=seifr_knn SEIFR_K_VALUES="${K_VALUES:-2 4 6 8 20}"

submit() {  # root env...
  local root=$1; shift
  env "$@" NODE_COUNTS=4 REPLICATION_GROUPS=1 EVAL_ROOT=$PWD/experiments/evals/$root \
    bash experiments/submit_full_evaluation.sh > experiments/logs/submit_$root.log 2>&1
  echo "$root: $(awk -F'\t' 'NR>1{print $4":"$9"("$6" runs)"}' experiments/evals/$root/submissions.tsv | tr '\n' ' ')"
}

submit seifrk_adaptive_$TAG ALGORITHMS="tremor dmass" TREMOR_REFINEMENT=adaptive
submit seifrk_sss_$TAG ALGORITHMS=tremor TREMOR_METHOD=skip-sequential
submit seifrk_dmassv3_$TAG ALGORITHMS=dmass MASS_BIN=$PWD/bin/dmass_v3_main
