# Shared list of seiFR waveform/template pairs for TREMOR and DMASS sweeps.
#
# This file is meant to be sourced from run_seifr_tremor.sh and
# run_seifr_dmass.sh. It defines:
#   - WAVEFORM_DIR : directory holding the seiFR waveform .bin files
#   - TEMPLATE_DIR : directory holding the seiFR template/query .bin files
#   - CASES        : array of "<waveform.bin> <queries.bin> <window_length>"
#
# SeiFR case files are discovered from generated binary names rather than
# hardcoded here. The Maule-style pipeline writes query binaries to
# template_bin_data; the legacy notebook output kept templates next to
# waveforms in bin_data, so this loader falls back to that location.

WAVEFORM_DIR="${SEIFR_WAVEFORM_CASE_DIR:-/path/to/data/seifr/waveforms}"
DEFAULT_TEMPLATE_DIR="/path/to/data/seifr/templates"
LEGACY_TEMPLATE_DIR="$WAVEFORM_DIR"
TEMPLATE_DIR="${SEIFR_TEMPLATE_CASE_DIR:-$DEFAULT_TEMPLATE_DIR}"
SEIFR_CASE_LIMIT="${SEIFR_CASE_LIMIT:-}"

CASES=()

discover_seifr_case_from_query() {
    local query_path="$1"
    local query_base waveform_key window_length

    query_base="$(basename "$query_path")"

    if [[ "$query_base" =~ ^tem[0-9]+\.([^.]+)\.len([0-9]+)\.fmin[0-9.]+\.fmax[0-9.]+\.bin$ ]]; then
        waveform_key="${BASH_REMATCH[1]}"
        window_length="${BASH_REMATCH[2]}"
    elif [[ "$query_base" =~ ^templ[0-9]+\.(.+)\.len([0-9]+)(\.sr_[0-9.]+)?\.bin$ ]]; then
        waveform_key="${BASH_REMATCH[1]}"
        window_length="${BASH_REMATCH[2]}"
    elif [[ "$query_base" =~ ^(.+)\.n[0-9]+\.len([0-9]+)\.bin$ ]]; then
        waveform_key="${BASH_REMATCH[1]}"
        window_length="${BASH_REMATCH[2]}"
    else
        return 1
    fi

    local waveform_matches=("$WAVEFORM_DIR/${waveform_key}.len"*.bin)
    if [[ ${#waveform_matches[@]} -eq 0 || ! -e "${waveform_matches[0]}" ]]; then
        echo "[!] Warning: no waveform match found for query $query_path" >&2
        return 1
    fi

    CASES+=("${waveform_matches[0]} $query_path $window_length")
    return 0
}

load_seifr_cases() {
    local query_path
    local template_dir
    local -a template_dirs
    local loaded_template_dir="$TEMPLATE_DIR"
    local loaded=0

    CASES=()
    template_dirs=("$TEMPLATE_DIR")
    if [[ -z "${SEIFR_TEMPLATE_CASE_DIR:-}" && "$LEGACY_TEMPLATE_DIR" != "$TEMPLATE_DIR" ]]; then
        template_dirs+=("$LEGACY_TEMPLATE_DIR")
    fi

    for template_dir in "${template_dirs[@]}"; do
        shopt -s nullglob
        for query_path in "$template_dir"/tem*.bin "$template_dir"/*.n*.len*.bin; do
            [[ -f "$query_path" ]] || continue
            if discover_seifr_case_from_query "$query_path"; then
                loaded=$((loaded + 1))
                loaded_template_dir="$template_dir"
                if [[ -n "$SEIFR_CASE_LIMIT" && "$loaded" -ge "$SEIFR_CASE_LIMIT" ]]; then
                    break
                fi
            fi
        done
        shopt -u nullglob

        if [[ ${#CASES[@]} -gt 0 ]]; then
            break
        fi
    done

    TEMPLATE_DIR="$loaded_template_dir"

    if [[ ${#CASES[@]} -eq 0 ]]; then
        echo "[!] No seiFR cases discovered." >&2
        echo "    WAVEFORM_DIR=$WAVEFORM_DIR" >&2
        echo "    TEMPLATE_DIR=$TEMPLATE_DIR" >&2
        if [[ -z "${SEIFR_TEMPLATE_CASE_DIR:-}" ]]; then
            echo "    LEGACY_TEMPLATE_DIR=$LEGACY_TEMPLATE_DIR" >&2
        fi
        return 1
    fi

    echo "[+] Loaded ${#CASES[@]} seiFR case(s) from $TEMPLATE_DIR"
}

load_seifr_cases
