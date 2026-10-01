#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/maule_datagen_job.sh"

MAULE_SDS_ROOT="${MAULE_SDS_ROOT:-/path/to/raw/maule/SDS/}"
MAULE_WAVEFORM_OUTPUT_DIR="${MAULE_WAVEFORM_OUTPUT_DIR:-/path/to/data/maule/waveforms/}"
MAULE_TEMPLATE_OUTPUT_DIR="${MAULE_TEMPLATE_OUTPUT_DIR:-/path/to/data/maule/templates/}"
MAULE_SUMMARY_OUTPUT_DIR="${MAULE_SUMMARY_OUTPUT_DIR:-$SCRIPT_DIR/metadata/}"
MAULE_SLURM_LOG_DIR="${MAULE_SLURM_LOG_DIR:-$SCRIPT_DIR/logs}"

# Set MAULE_WAVEFORM_ARRAY="" before running this wrapper for one non-array
# waveform job. The default below follows the array split documented in
# maule_datagen_job.sh.
MAULE_WAVEFORM_ARRAY="${MAULE_WAVEFORM_ARRAY-0-23}"

MAULE_GENERATION_CHUNK_DAYS="${MAULE_GENERATION_CHUNK_DAYS:-1}"
MAULE_FILTER_FMIN="${MAULE_FILTER_FMIN:-1}"
MAULE_FILTER_FMAX="${MAULE_FILTER_FMAX:-10}"
MAULE_DOWNSAMPLE_FACTOR="${MAULE_DOWNSAMPLE_FACTOR:-2}"
MAULE_ENABLE_WAVEFORM_REPORT="${MAULE_ENABLE_WAVEFORM_REPORT:-1}"
MAULE_ENABLE_TEMPLATE_REPORT="${MAULE_ENABLE_TEMPLATE_REPORT:-1}"
MAULE_WAVEFORM_PLOT_MAX_POINTS="${MAULE_WAVEFORM_PLOT_MAX_POINTS:-200000}"
MAULE_WAVEFORM_PLOT_POINTS_PER_CHUNK="${MAULE_WAVEFORM_PLOT_POINTS_PER_CHUNK:-4000}"
MAULE_TEMPLATE_REPORT_ROW_PREVIEW_LIMIT="${MAULE_TEMPLATE_REPORT_ROW_PREVIEW_LIMIT:-20}"
MAULE_TEMPLATE_PLOT_SAMPLE_COUNT="${MAULE_TEMPLATE_PLOT_SAMPLE_COUNT:-5}"
MAULE_TEMPLATE_PLOT_SEED="${MAULE_TEMPLATE_PLOT_SEED:-0}"

MAULE_SUBMIT_WAVEFORMS="${MAULE_SUBMIT_WAVEFORMS:-1}"
MAULE_SUBMIT_TEMPLATES="${MAULE_SUBMIT_TEMPLATES:-1}"
MAULE_TEMPLATES_AFTER_WAVEFORMS="${MAULE_TEMPLATES_AFTER_WAVEFORMS:-1}"

MAULE_WAVEFORM_JOB_NAME="${MAULE_WAVEFORM_JOB_NAME:-maule-waveforms}"
MAULE_TEMPLATE_JOB_NAME="${MAULE_TEMPLATE_JOB_NAME:-maule-templates}"

mkdir -p \
    "$MAULE_SLURM_LOG_DIR" \
    "$MAULE_WAVEFORM_OUTPUT_DIR" \
    "$MAULE_TEMPLATE_OUTPUT_DIR" \
    "$MAULE_SUMMARY_OUTPUT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch was not found in PATH. Run this wrapper on a Slurm login node." >&2
    exit 1
fi

COMMON_EXPORTS=(
    "MAULE_SDS_ROOT=$MAULE_SDS_ROOT"
    "MAULE_WAVEFORM_OUTPUT_DIR=$MAULE_WAVEFORM_OUTPUT_DIR"
    "MAULE_TEMPLATE_OUTPUT_DIR=$MAULE_TEMPLATE_OUTPUT_DIR"
    "MAULE_SUMMARY_OUTPUT_DIR=$MAULE_SUMMARY_OUTPUT_DIR"
    "MAULE_SLURM_LOG_DIR=$MAULE_SLURM_LOG_DIR"
    "MAULE_GENERATION_CHUNK_DAYS=$MAULE_GENERATION_CHUNK_DAYS"
    "MAULE_FILTER_FMIN=$MAULE_FILTER_FMIN"
    "MAULE_FILTER_FMAX=$MAULE_FILTER_FMAX"
    "MAULE_DOWNSAMPLE_FACTOR=$MAULE_DOWNSAMPLE_FACTOR"
    "MAULE_ENABLE_WAVEFORM_REPORT=$MAULE_ENABLE_WAVEFORM_REPORT"
    "MAULE_ENABLE_TEMPLATE_REPORT=$MAULE_ENABLE_TEMPLATE_REPORT"
    "MAULE_WAVEFORM_PLOT_MAX_POINTS=$MAULE_WAVEFORM_PLOT_MAX_POINTS"
    "MAULE_WAVEFORM_PLOT_POINTS_PER_CHUNK=$MAULE_WAVEFORM_PLOT_POINTS_PER_CHUNK"
    "MAULE_TEMPLATE_REPORT_ROW_PREVIEW_LIMIT=$MAULE_TEMPLATE_REPORT_ROW_PREVIEW_LIMIT"
    "MAULE_TEMPLATE_PLOT_SAMPLE_COUNT=$MAULE_TEMPLATE_PLOT_SAMPLE_COUNT"
    "MAULE_TEMPLATE_PLOT_SEED=$MAULE_TEMPLATE_PLOT_SEED"
)

append_if_set() {
    local name="$1"
    local value="${!name:-}"
    if [[ -n "$value" ]]; then
        COMMON_EXPORTS+=("$name=$value")
    fi
}

append_if_set MAULE_GENERATION_MAX_DAYS
append_if_set MAULE_MAX_GENERATION_ENTRIES
append_if_set MAULE_TEMPLATE_CHUNKSIZE
append_if_set MAULE_TEMPLATE_LOG_LIMIT

join_by_comma() {
    local IFS=,
    echo "$*"
}

make_export_arg() {
    echo "ALL,$(join_by_comma "$@")"
}

echo "Submitting Maule generation jobs"
echo "  Slurm logs: $MAULE_SLURM_LOG_DIR"
echo "  Waveform output: $MAULE_WAVEFORM_OUTPUT_DIR"
echo "  Template output: $MAULE_TEMPLATE_OUTPUT_DIR"
echo "  Summary output: $MAULE_SUMMARY_OUTPUT_DIR"
echo "  Waveform array: ${MAULE_WAVEFORM_ARRAY:-none}"
echo "  Filter/downsample: ${MAULE_FILTER_FMIN}-${MAULE_FILTER_FMAX} Hz, factor $MAULE_DOWNSAMPLE_FACTOR"

waveform_job_id=""
if [[ "$MAULE_SUBMIT_WAVEFORMS" == "1" ]]; then
    WAVEFORM_EXPORTS=(
        "${COMMON_EXPORTS[@]}"
        "RUN_WAVEFORMS=1"
        "RUN_TEMPLATES=0"
        "MAULE_RUN_LABEL=waveforms"
    )
    WAVEFORM_SBATCH_ARGS=(
        --parsable
        --job-name="$MAULE_WAVEFORM_JOB_NAME"
        --output="$MAULE_SLURM_LOG_DIR/$MAULE_WAVEFORM_JOB_NAME.%A_%a.out"
        --error="$MAULE_SLURM_LOG_DIR/$MAULE_WAVEFORM_JOB_NAME.%A_%a.err"
        --export="$(make_export_arg "${WAVEFORM_EXPORTS[@]}")"
    )
    if [[ -n "$MAULE_WAVEFORM_ARRAY" ]]; then
        WAVEFORM_SBATCH_ARGS+=(--array="$MAULE_WAVEFORM_ARRAY")
    fi

    waveform_job_id="$(sbatch "${WAVEFORM_SBATCH_ARGS[@]}" "$JOB_SCRIPT")"
    echo "Submitted waveform job: $waveform_job_id"
    echo "  stdout pattern: $MAULE_SLURM_LOG_DIR/$MAULE_WAVEFORM_JOB_NAME.%A_%a.out"
    echo "  stderr pattern: $MAULE_SLURM_LOG_DIR/$MAULE_WAVEFORM_JOB_NAME.%A_%a.err"
else
    echo "Skipping waveform submission because MAULE_SUBMIT_WAVEFORMS=$MAULE_SUBMIT_WAVEFORMS"
fi

if [[ "$MAULE_SUBMIT_TEMPLATES" == "1" ]]; then
    TEMPLATE_EXPORTS=(
        "${COMMON_EXPORTS[@]}"
        "RUN_WAVEFORMS=0"
        "RUN_TEMPLATES=1"
        "MAULE_RUN_LABEL=templates"
    )
    TEMPLATE_SBATCH_ARGS=(
        --parsable
        --job-name="$MAULE_TEMPLATE_JOB_NAME"
        --output="$MAULE_SLURM_LOG_DIR/$MAULE_TEMPLATE_JOB_NAME.%A.out"
        --error="$MAULE_SLURM_LOG_DIR/$MAULE_TEMPLATE_JOB_NAME.%A.err"
        --export="$(make_export_arg "${TEMPLATE_EXPORTS[@]}")"
    )
    if [[ "$MAULE_TEMPLATES_AFTER_WAVEFORMS" == "1" && -n "$waveform_job_id" ]]; then
        TEMPLATE_SBATCH_ARGS+=(--dependency="afterok:$waveform_job_id")
        echo "Template job will wait for waveform job $waveform_job_id"
    fi

    template_job_id="$(sbatch "${TEMPLATE_SBATCH_ARGS[@]}" "$JOB_SCRIPT")"
    echo "Submitted template job: $template_job_id"
    echo "  stdout pattern: $MAULE_SLURM_LOG_DIR/$MAULE_TEMPLATE_JOB_NAME.%A.out"
    echo "  stderr pattern: $MAULE_SLURM_LOG_DIR/$MAULE_TEMPLATE_JOB_NAME.%A.err"
else
    echo "Skipping template submission because MAULE_SUBMIT_TEMPLATES=$MAULE_SUBMIT_TEMPLATES"
fi

echo "Done submitting Maule generation jobs."
