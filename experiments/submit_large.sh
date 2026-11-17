#!/bin/bash
# The query experiments of the paper, on the large data:
#   Maule: every channel with templates that one node can hold for every method (<= 2.1G points: 243 of the 249
#          channels, 166.5G points, 92% of the dataset; the 6 channels of 2.5G points exceed DMASS's FFT length limit
#          and TREMOR's index under full replication), 50 templates per channel (11,922; sampled with a seed per
#          channel), threshold search;
#   SeiFR: the 7 channels with templates, ALL 4,430 templates, k-NN search.
# Kinds:
#   main:  4 nodes, full replication, Maule t = 0.6 .. 0.999, SeiFR k = 1, 2, 5, 10
#   scale: 1, 2, 8 nodes, full replication, Maule t = 0.9, SeiFR k = 5
#   repl4: 4 nodes, replication groups 0 and 2 (no replication, Partial-2), Maule t = 0.9, SeiFR k = 5
#   none8: 8 nodes without replication, Maule t = 0.9, SeiFR k = 5 (TREMOR only: its index construction on 1 and 8 nodes,
#          Section 4.2.5)
# Methods: tremor, sss (bin/tremor_main), dmassv3, dmassv1 (bin/dmass_v3_main, bin/dmass_main), one run (one index
# build or preprocessing) per setting. DMASS-V1 gets a lower priority (nice) than the other methods.
# Roots large_<dataset>_<method>_<kind>_<TAG>; roots that were already submitted are skipped.
# Usage: TAG=20261005 [METHODS="tremor sss dmassv3 dmassv1"] [KINDS="main scale repl4"] [DATASETS="maule seifr"] \
#        bash experiments/submit_large.sh
set -euo pipefail
cd /path/to/TREMOR
TAG=${TAG:?set TAG}
source experiments/fullrepl_env.sh
export LAUNCH_WRAPPER="numactl --interleave=all" PAA_BY_WINDOW="1000=10 992=16 496=16 3000=8"
export MAULE_CASE_COUNT=0 MAULE_MAX_SAMPLES=2100000000 ARRAY_THROTTLE=${ARRAY_THROTTLE:-4}
export SUBMIT_DEPENDENCY=${DEPENDENCY:-}

submit() {  # root nodes groups env...
  local root=$1 nodes=$2 groups=$3; shift 3
  [[ -f experiments/evals/$root/submissions.tsv ]] && { echo "$root: already submitted"; return; }
  env "$@" NODE_COUNTS="$nodes" REPLICATION_GROUPS="$groups" EVAL_ROOT=$PWD/experiments/evals/$root \
    bash experiments/submit_full_evaluation.sh > experiments/logs/submit_$root.log 2>&1
  echo "$root: $(awk -F'\t' 'NR>1{print $2"_"$3".n"$5":"$9"("$6" tasks, "$7" runs)"}' experiments/evals/$root/submissions.tsv | tr '\n' ' ')"
}

# Row timeouts (s) per node count, about 3x the longest run expected (from runs on 20 Maule channels; SeiFR: scaled to
# all templates).
timeouts() {  # dataset method
  case $1.$2 in
    maule.tremor|maule.sss|maule.dmassv3) echo "1=1800 2=1200 4=1200 8=1200" ;;
    maule.dmassv1) echo "1=7200 2=3600 4=1800 8=1200" ;;
    seifr.tremor) echo "1=2400 2=1800 4=1800 8=1800" ;;
    seifr.sss) echo "1=14400 2=7200 4=4200 8=2400" ;;
    seifr.dmassv3) echo "1=6000 2=3600 4=2400 8=1800" ;;
    seifr.dmassv1) echo "1=57600 2=32400 4=19800 8=10800" ;;
  esac
}

for kind in ${KINDS:-main scale repl4}; do
  for dataset in ${DATASETS:-maule seifr}; do
    case $dataset in
      maule) data=(WORKLOADS=maule_threshold QUERY_LIMIT=50); main=(MAULE_THRESHOLDS="0.6 0.7 0.8 0.9 0.95 0.99 0.999"); one=(MAULE_THRESHOLDS=0.9); ;;
      seifr) data=(WORKLOADS=seifr_knn QUERY_LIMIT=0); main=(SEIFR_K_VALUES="1 2 5 10"); one=(SEIFR_K_VALUES=5); ;;
    esac
    case $kind in
      main)  nodes=4; groups=1; grid=("${main[@]}") ;;
      scale) nodes="1 2 8"; groups=1; grid=("${one[@]}") ;;
      repl4) nodes=4; groups="0 2"; grid=("${one[@]}") ;;
      none8) nodes=8; groups=0; grid=("${one[@]}") ;;
    esac
    for method in ${METHODS:-tremor sss dmassv3 dmassv1}; do
      case $method in
        tremor)  env=(ALGORITHMS=tremor TREMOR_REFINEMENT=adaptive) ;;
        sss)     env=(ALGORITHMS=tremor TREMOR_METHOD=skip-sequential) ;;
        dmassv3) env=(ALGORITHMS=dmass MASS_BIN=$PWD/bin/dmass_v3_main) ;;
        dmassv1) env=(ALGORITHMS=dmass MASS_BIN=$PWD/bin/dmass_main SBATCH_NICE=200) ;;
      esac
      # Runs per array task: several for the short Maule runs (main: the 7 thresholds of a channel), one for the long
      # SeiFR baselines.
      if [[ $dataset = maule ]]; then per_task=$([[ $kind = main ]] && echo 7 || echo 4)
      elif [[ $method =~ ^dmass ]]; then per_task=1
      else per_task=2
      fi
      submit large_${dataset}_${method}_${kind}_$TAG "$nodes" "$groups" "${data[@]}" "${grid[@]}" "${env[@]}" \
        RUNS_PER_ARRAY_TASK=$per_task ROW_TIMEOUT_BY_NODES="$(timeouts $dataset $method)"
    done
  done
done
