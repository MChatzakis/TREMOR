#!/bin/bash
#SBATCH --job-name=tremor-tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=experiments/logs/tremor-tuning.%j.out
#SBATCH --error=experiments/logs/tremor-tuning.%j.err
#
# TREMOR sensitivity to PAA segments (and search workers) on inputs from the
# scaling evaluation. Each output is compared with the finished DMASS n1 result
# for the same case and queries, so a faster setting is only accepted if exact.
# TUNING_SET=maule|seifr selects the cases.
set -uo pipefail
cd /path/to/TREMOR
module load intel/21U2/suite
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export OMP_NUM_THREADS=128
PY=python3
EV=experiments/evals/scaling_20260924
OUT="experiments/evals/tremor_tuning_${SLURM_JOB_ID}"
mkdir -p "$OUT"
cp bin/tremor_main "$OUT/"

# run <name> <case_id> <waveform> <samples> <queries> <nq> <window> <mode> <paa> <workers>
run() {
    local name=$1 case=$2 wave=$3 n=$4 q=$5 nq=$6 w=$7 mode=$8 paa=$9 workers=${10}
    local -a extra
    if [[ $mode = knn ]]; then
        extra=(--k 10 --merge-offset 0 --output "$OUT/$name.csv")
    else
        extra=(--threshold 0.8 --threshold-result-slots 1 --merge-offset $((w / 4)) --output "$OUT/$name")
    fi
    echo "=== $name paa=$paa workers=$workers"
    local t0=$SECONDS
    srun --mpi=pmi2 -n 1 -c 128 "$OUT/tremor_main" --dataset "$wave" --dataset-size "$n" --queries "$q" \
        --query-count "$nq" --window-length "$w" --mode "$mode" --replication-groups 0 \
        --index-threads 128 --search-workers "$workers" --leaf-size 2000 --paa-segments "$paa" \
        --pq-threshold-divisor 16 "${extra[@]}" --benchmark-output "$OUT/$name.timings.csv" > "$OUT/$name.log" 2>&1
    echo "$name exit=$? wall=$((SECONDS - t0))s timings: $(tail -n +2 "$OUT/$name.timings.csv" 2>/dev/null)"
}

MW=/path/to/data/maule/waveforms
SW=/path/to/data/seifr/waveforms
if [[ "${TUNING_SET:-maule}" = maule ]]; then
    GA=(ZE_GA1S__HHE $MW/ZE_GA1S__HHE.len87316288.bin 87316288 $EV/inputs/maule/ZE_GA1S__HHE.q50.bin 50 1000 threshold)
    run GA1S.paa8 "${GA[@]}" 8 128
    run GA1S.paa8.w32 "${GA[@]}" 8 32
    run GA1S.paa10 "${GA[@]}" 10 128
    run GA1S.paa20 "${GA[@]}" 20 128
    GC=(ZE_GC2S__HHN $MW/ZE_GC2S__HHN.len613697643.bin 613697643 $EV/inputs/maule/ZE_GC2S__HHN.q50.bin 50 992 threshold)
    run GC2S.paa16 "${GC[@]}" 16 128
else
    EP=(RD_EPF__BHZ $SW/RD_EPF__BHZ.len1046593001.bin 1046593001 $EV/inputs/seifr/RD_EPF__BHZ.q50.bin 10 3000 knn)
    run EPF.paa8 "${EP[@]}" 8 128
    run EPF.paa12 "${EP[@]}" 12 128
    run EPF.paa20 "${EP[@]}" 20 128
    run EPF.paa24 "${EP[@]}" 24 128
fi

# Exactness: every setting must reproduce the DMASS result for the same queries.
$PY - "$OUT" "$EV" <<'EOF'
import csv, glob, os, sys
out, ev = sys.argv[1], sys.argv[2]
def rows(paths, key):
    s = set()
    for p in paths:
        for r in list(csv.reader(open(p)))[1:]:
            r = [x.strip() for x in r]
            if int(r[0]) < key: s.add((int(r[0]), int(r[2])))
    return s
ref = {
    'GA1S': (glob.glob(f'{ev}/results/maule_threshold/dmass/maule_threshold.ZE_GA1S__HHE.*.n1.r0_*.csv'), 50),
    'GC2S': (glob.glob(f'{ev}/results/maule_threshold/dmass/maule_threshold.ZE_GC2S__HHN.*.n1.r0_*.csv'), 50),
    'EPF': ([f'{ev}/results/seifr_knn/dmass/seifr_knn.RD_EPF__BHZ.w3000.q50.k10.n1.r0.csv'], 10),
}
for t in sorted(glob.glob(f'{out}/*.timings.csv')):
    name = os.path.basename(t)[:-len('.timings.csv')]
    case = name.split('.')[0]
    paths, nq = ref[case]
    mine = glob.glob(f'{out}/{name}_*.csv') or [f'{out}/{name}.csv']
    a, b = rows(mine, nq), rows(paths, nq)
    print(f'{name}: tremor {len(a)} dmass {len(b)} -> {"MATCH" if a == b else "MISMATCH"}')
EOF
