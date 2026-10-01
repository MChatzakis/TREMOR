#!/bin/bash
# The evaluation of the paper on subsets of the datasets (Maule: 20 channels stratified by length among those DMASS can
# hold on one node, 56 GB; SeiFR: the 7 channels with templates, 42 GB; 50 templates per channel). Kinds of runs:
#   main:  4 nodes, full replication, t = 0.6 .. 0.999 and k = 1, 2, 5, 10
#   scale: 1, 2, 8 nodes, full replication, t = 0.8 and k = 5
#   repl8: 8 nodes, replication groups 0/2/4 (none, partial-4, partial-2), t = 0.8 and k = 5
#   repl4: 4 nodes, replication groups 0/2 (none, partial-2), t = 0.8 and k = 5 (full replication is in main)
#   scalet09, repl4t09: as scale and repl4, for t = 0.9 (threshold workloads only)
#   scalek5, repl4k5: as scale and repl4, for k = 5 only (k-NN workloads only)
#   main8, repl8t09, repl8k5: 8 nodes as the default (the threshold and k sweeps, and the replication degrees)
#   index: Maule, 4 nodes, full replication, t = 0.99, 0.999 and every k: the settings where TREMOR uses its index
# The paper varies the nodes and the replication degree for t = 0.9 (Maule) and k = 5 (SeiFR).
# Methods: tremor, sss (skip-sequential scan), ss (sequential scan without lower bounds), dmassv1, dmassv3.
# PAPER_ONLY=1: only the workloads of the paper (Maule threshold search, SeiFR k-NN search).
# Roots experiments/evals/sub_<method>_<kind>_<TAG>; roots that were already submitted are skipped. The notebook
# (notebooks/analysis.ipynb) reads the roots of TAG=20260929.
# Usage: TAG=20260929 METHODS="dmassv1 dmassv3 sss tremor" KINDS="main scale repl4 repl8 scalet09 repl4t09" PAPER_ONLY=1 \
#        [DEPENDENCY=...] bash experiments/submit_subsets.sh
set -euo pipefail
cd /path/to/TREMOR
TAG=${TAG:?set TAG}
source experiments/fullrepl_env.sh
export LAUNCH_WRAPPER="numactl --interleave=all" PAA_BY_WINDOW="1000=10 992=16 496=16 3000=8"
export MAULE_CASE_COUNT=20 MAULE_MAX_SAMPLES=2100000000 QUERY_LIMIT=50 RUNS_PER_ARRAY_TASK=1 ARRAY_THROTTLE=4
export WORKLOADS="maule_threshold maule_knn seifr_threshold seifr_knn" SUBMIT_DEPENDENCY=${DEPENDENCY:-}
export ROW_TIMEOUT_BY_NODES="1=5400 2=3600 4=1800 8=1200"  # short limits: easier backfill on a busy cluster

submit() {  # root nodes groups env...
  local root=$1 nodes=$2 groups=$3; shift 3
  [[ -f experiments/evals/$root/submissions.tsv ]] && { echo "$root: already submitted"; return; }
  env "$@" NODE_COUNTS="$nodes" REPLICATION_GROUPS="$groups" EVAL_ROOT=$PWD/experiments/evals/$root \
    bash experiments/submit_full_evaluation.sh > experiments/logs/submit_$root.log 2>&1
  echo "$root: $(awk -F'\t' 'NR>1{print $2"_"$3".n"$5":"$9"("$7")"}' experiments/evals/$root/submissions.tsv | tr '\n' ' ')"
}

for kind in ${KINDS:-main scale repl8}; do
  case $kind in
    main)  nodes=4; groups=1; grid=(MAULE_THRESHOLDS="0.6 0.7 0.8 0.9 0.95 0.99 0.999" MAULE_K_VALUES="1 2 5 10" SEIFR_K_VALUES="1 2 5 10") ;;
    scale) nodes="1 2 8"; groups=1; grid=(MAULE_THRESHOLDS=0.8 MAULE_K_VALUES=5 SEIFR_K_VALUES=5) ;;
    repl8) nodes=8; groups="0 2 4"; grid=(MAULE_THRESHOLDS=0.8 MAULE_K_VALUES=5 SEIFR_K_VALUES=5) ;;
    repl4) nodes=4; groups="0 2"; grid=(MAULE_THRESHOLDS=0.8 MAULE_K_VALUES=5 SEIFR_K_VALUES=5) ;;
    scalet09) nodes="1 2 8"; groups=1; grid=(MAULE_THRESHOLDS=0.9 WORKLOADS="maule_threshold seifr_threshold") ;;
    repl4t09) nodes=4; groups="0 2"; grid=(MAULE_THRESHOLDS=0.9 WORKLOADS="maule_threshold seifr_threshold") ;;
    scalek5) nodes="1 2 8"; groups=1; grid=(MAULE_K_VALUES=5 SEIFR_K_VALUES=5 WORKLOADS="maule_knn seifr_knn") ;;
    repl4k5) nodes=4; groups="0 2"; grid=(MAULE_K_VALUES=5 SEIFR_K_VALUES=5 WORKLOADS="maule_knn seifr_knn") ;;
    # 8 nodes as the default: the rest of the threshold / k sweep (t = 0.8, 0.9 and k = 5 are in scale and scalet09),
    # and the replication degrees for t = 0.9 and k = 5
    main8) nodes=8; groups=1; grid=(MAULE_THRESHOLDS="0.6 0.7 0.95 0.99 0.999" SEIFR_K_VALUES="1 2 10" WORKLOADS="maule_threshold seifr_knn") ;;
    repl8t09) nodes=8; groups="0 2 4"; grid=(MAULE_THRESHOLDS=0.9 WORKLOADS=maule_threshold) ;;
    repl8k5) nodes=8; groups="0 2 4"; grid=(SEIFR_K_VALUES=5 WORKLOADS=seifr_knn) ;;
    index) nodes=4; groups=1; grid=(MAULE_THRESHOLDS="0.99 0.999" MAULE_K_VALUES="1 2 5 10" WORKLOADS="maule_threshold maule_knn") ;;
  esac
  if [[ -n ${PAPER_ONLY:-} ]]; then  # the last WORKLOADS assignment wins
    case $kind in
      main) grid+=(WORKLOADS="maule_threshold seifr_knn") ;;
      scalet09|repl4t09) grid+=(WORKLOADS=maule_threshold) ;;
      scalek5|repl4k5) grid+=(WORKLOADS=seifr_knn) ;;
    esac
  fi
  for method in ${METHODS:-dmassv1 dmassv3 sss tremor}; do
    case $method in
      tremor)  env=(ALGORITHMS=tremor TREMOR_REFINEMENT=adaptive) ;;
      sss)     env=(ALGORITHMS=tremor TREMOR_METHOD=skip-sequential) ;;
      ss)      env=(ALGORITHMS=tremor TREMOR_METHOD=sequential) ;;
      dmassv1) env=(ALGORITHMS=dmass MASS_BIN=$PWD/bin/dmass_main) ;;
      dmassv3) env=(ALGORITHMS=dmass MASS_BIN=$PWD/bin/dmass_v3_main) ;;
    esac
    submit sub_${method}_${kind}_$TAG "$nodes" "$groups" "${grid[@]}" "${env[@]}"
  done
done
