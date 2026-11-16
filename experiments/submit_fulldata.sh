#!/bin/bash
# Full-data comparison on 4 nodes with full replication: every Maule channel with templates (249 channels,
# 727 GB; 50 templates per channel) at t = 0.8, 0.9, 0.99, and every SeiFR channel with templates (the 7
# vertical channels; all 4,430 templates) at k = 1, 10. Added later: Maule t = 0.999 (roots fulldata_maule_<method>_<TAG>b)
# and t = 0.6, 0.7 (..._<TAG>c), SeiFR k = 2, 5, 20 (fulldata_seifr_<method>_<TAG>b). Methods: TREMOR, SSS, SS (bin/tremor_main) and
# DMASS-V3 (bin/dmass_v3_main); DMASS-V1 runs only on the subset (it is ~10x slower than DMASS-V3).
# Roots fulldata_<dataset>_<method>_<TAG>; roots that were already submitted are skipped.
# Usage: TAG=20260929 [DEPENDENCY=afterok:<job>] bash experiments/submit_fulldata.sh
set -euo pipefail
cd /path/to/TREMOR
TAG=${TAG:?set TAG}
source experiments/fullrepl_env.sh
export LAUNCH_WRAPPER="numactl --interleave=all" PAA_BY_WINDOW="1000=10 992=16 496=16 3000=8"
export MAULE_CASE_COUNT=0 RUNS_PER_ARRAY_TASK=3 ARRAY_THROTTLE=3
export SUBMIT_DEPENDENCY=${DEPENDENCY:-}

submit() {  # root env...
  local root=$1; shift
  [[ -f experiments/evals/$root/submissions.tsv ]] && { echo "$root: already submitted"; return; }
  env "$@" NODE_COUNTS=4 REPLICATION_GROUPS=1 EVAL_ROOT=$PWD/experiments/evals/$root \
    bash experiments/submit_full_evaluation.sh > experiments/logs/submit_$root.log 2>&1
  echo "$root: $(awk -F'\t' 'NR>1{print $4":"$9"("$7" runs)"}' experiments/evals/$root/submissions.tsv | tr '\n' ' ')"
}

MAULE=(WORKLOADS=maule_threshold MAULE_THRESHOLDS="0.8 0.9 0.99" QUERY_LIMIT=50 ROW_TIMEOUT_BY_NODES="4=1800")
MAULE999=(WORKLOADS=maule_threshold MAULE_THRESHOLDS=0.999 QUERY_LIMIT=50 ROW_TIMEOUT_BY_NODES="4=1800")  # added later (root suffix b)
MAULE067=(WORKLOADS=maule_threshold MAULE_THRESHOLDS="0.6 0.7" QUERY_LIMIT=50 ROW_TIMEOUT_BY_NODES="4=2400")  # root suffix c
SEIFRK=(WORKLOADS=seifr_knn SEIFR_K_VALUES="2 5 20" QUERY_LIMIT=5000 ROW_TIMEOUT_BY_NODES="4=7200")  # root suffix b
SEIFR=(WORKLOADS=seifr_knn SEIFR_K_VALUES="1 10" QUERY_LIMIT=5000 ROW_TIMEOUT_BY_NODES="4=7200")
for data in maule seifr maule999 maule067 seifrk; do
  case $data in maule) grid=("${MAULE[@]}"); root=maule_TAG ;; seifr) grid=("${SEIFR[@]}"); root=seifr_TAG ;;
                maule999) grid=("${MAULE999[@]}"); root=maule_TAGb ;; maule067) grid=("${MAULE067[@]}"); root=maule_TAGc ;;
                seifrk) grid=("${SEIFRK[@]}"); root=seifr_TAGb ;; esac
  for method in tremor sss ss dmassv3; do
    case $method in tremor) env=(ALGORITHMS=tremor TREMOR_REFINEMENT=adaptive) ;; sss) env=(ALGORITHMS=tremor TREMOR_METHOD=skip-sequential) ;;
                    ss) env=(ALGORITHMS=tremor TREMOR_METHOD=sequential) ;; dmassv3) env=(ALGORITHMS=dmass MASS_BIN=$PWD/bin/dmass_v3_main) ;; esac
    submit fulldata_${root/_TAG/_${method}_$TAG} "${grid[@]}" "${env[@]}"
  done
done
