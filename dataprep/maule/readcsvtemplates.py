from obspy import UTCDateTime
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality import PIPELINE_VERSION, complete_template

from shared_pipeline import (
    DOWNSAMPLE_FACTOR,
    FMAX,
    FMIN,
    SDS_ROOT,
    SELECT_CHANNELS,
    SELECT_LOCATIONS,
    SELECT_NETWORKS,
    SELECT_STATIONS,
    TEMPLATE_CHUNKSIZE,
    TEMPLATE_LOG_LIMIT,
    ensure_parent_directory,
    get_trace_downsample_factor,
    get_trace_original_sampling_rate,
    load_and_prepare_trace,
    matches_selection,
    tqdm,
)

DIRECTORY = os.getenv(
    "MAULE_TEMPLATE_OUTPUT_DIR",
    "/path/to/data/maule/templates/",
)

_FALSE_ENV_VALUES = {"0", "false", "no", "off", ""}


def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


# Toggle the per-template-binary JSON metadata sidecars.
ENABLE_TEMPLATE_GENERATION_REPORT = env_flag("MAULE_ENABLE_TEMPLATE_REPORT", True)
SUMMARY_OUTPUT_DIR = Path(
    os.getenv(
        "MAULE_SUMMARY_OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "metadata"),
    )
)
TEMPLATE_REPORT_ROW_PREVIEW_LIMIT = int(
    os.getenv("MAULE_TEMPLATE_REPORT_ROW_PREVIEW_LIMIT", "20")
)
TEMPLATE_PLOT_SAMPLE_COUNT = int(
    os.getenv("MAULE_TEMPLATE_PLOT_SAMPLE_COUNT", "5")
)
TEMPLATE_PLOT_SEED = int(os.getenv("MAULE_TEMPLATE_PLOT_SEED", "0"))

# A handful of picks in templates_all_stations_channels.csv have their endtime
# clamped to the last sample of the day (day-boundary clipping upstream), which
# yields windows a few seconds instead of ~10s. Drop those rather than let one
# short window collapse an entire station/channel's templates to its length.
NOMINAL_QUERY_DURATION_SECONDS = float(
    os.getenv("MAULE_NOMINAL_QUERY_DURATION_SECONDS", "10.0")
)
QUERY_DURATION_TOLERANCE_SECONDS = float(
    os.getenv("MAULE_QUERY_DURATION_TOLERANCE_SECONDS", "1.0")
)


def save_binary_file(filename, data):
    ensure_parent_directory(filename)
    data = np.asarray(data, dtype=np.float32)
    print(f">> BinFile: Writing {data.size} points to {filename}")
    data.tofile(filename)


def pad_to_mul(num, mul=8):
    if mul == 0:
        return num

    remainder = num % mul
    if remainder == 0:
        return num

    assert (num - remainder) % mul == 0

    return num - remainder


def make_summary_file_path(binary_path, suffix):
    return SUMMARY_OUTPUT_DIR / Path(binary_path).with_suffix(suffix).name


def utc_datetime_to_string(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def current_utc_timestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def selected_values(values):
    return sorted(values) if values is not None else None


def write_json_file(filename, data):
    ensure_parent_directory(filename)
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)


def make_template_group_metadata():
    return {
        "template_count": 0,
        "csv_row_indices_preview": [],
        "requested_start_min": None,
        "requested_end_max": None,
        "trace_start_min": None,
        "trace_end_max": None,
        "original_sampling_rates_hz": set(),
        "sampling_rates_hz": set(),
        "downsample_factor_counts": Counter(),
        "original_length_counts": Counter(),
    }


def update_min_time(current, value):
    if value is None:
        return current
    return value if current is None else min(current, value)


def update_max_time(current, value):
    if value is None:
        return current
    return value if current is None else max(current, value)


def update_template_group_metadata(
    template_group_metadata,
    key,
    row_index,
    requested_start,
    requested_end,
    template_trace,
):
    group = template_group_metadata.setdefault(key, make_template_group_metadata())
    group["template_count"] += 1
    if len(group["csv_row_indices_preview"]) < TEMPLATE_REPORT_ROW_PREVIEW_LIMIT:
        group["csv_row_indices_preview"].append(int(row_index))

    group["requested_start_min"] = update_min_time(
        group["requested_start_min"], requested_start
    )
    group["requested_end_max"] = update_max_time(
        group["requested_end_max"], requested_end
    )
    group["trace_start_min"] = update_min_time(
        group["trace_start_min"], template_trace.stats.starttime
    )
    group["trace_end_max"] = update_max_time(
        group["trace_end_max"], template_trace.stats.endtime
    )
    group["original_sampling_rates_hz"].add(
        get_trace_original_sampling_rate(template_trace)
    )
    group["sampling_rates_hz"].add(float(template_trace.stats.sampling_rate))
    group["downsample_factor_counts"][
        get_trace_downsample_factor(template_trace)
    ] += 1
    group["original_length_counts"][int(len(template_trace.data))] += 1


def finalize_template_group_metadata(group):
    if not group:
        return {}

    return {
        "template_count": group["template_count"],
        "csv_row_indices_preview": group["csv_row_indices_preview"],
        "csv_row_indices_preview_limit": TEMPLATE_REPORT_ROW_PREVIEW_LIMIT,
        "requested_start_min": utc_datetime_to_string(group["requested_start_min"]),
        "requested_end_max": utc_datetime_to_string(group["requested_end_max"]),
        "trace_start_min": utc_datetime_to_string(group["trace_start_min"]),
        "trace_end_max": utc_datetime_to_string(group["trace_end_max"]),
        "original_sampling_rates_hz": sorted(group["original_sampling_rates_hz"]),
        "sampling_rates_hz": sorted(group["sampling_rates_hz"]),
        "downsample_factor_counts": [
            {"factor": factor, "count": count}
            for factor, count in sorted(group["downsample_factor_counts"].items())
        ],
        "original_length_counts": [
            {"length": length, "count": count}
            for length, count in sorted(group["original_length_counts"].items())
        ],
    }


def array_stats(data):
    if data.size == 0:
        return {
            "finite_points": 0,
            "nan_points": 0,
            "inf_points": 0,
            "min": None,
            "max": None,
            "mean": None,
        }

    finite_mask = np.isfinite(data)
    finite_points = int(finite_mask.sum())
    finite_data = data if finite_points == data.size else data[finite_mask]
    return {
        "finite_points": finite_points,
        "nan_points": int(np.isnan(data).sum()),
        "inf_points": int(np.isinf(data).sum()),
        "min": float(finite_data.min()) if finite_points else None,
        "max": float(finite_data.max()) if finite_points else None,
        "mean": (
            float(finite_data.sum(dtype=np.float64) / finite_points)
            if finite_points
            else None
        ),
    }


def write_template_sample_plot(
    binary_path,
    all_template_data,
    number_of_templates,
    template_length,
    network,
    station,
    location,
    channel,
):
    """Randomly sample a handful of packed templates and render them into a
    single PNG with one subplot per sampled template."""

    if TEMPLATE_PLOT_SAMPLE_COUNT <= 0 or number_of_templates <= 0:
        return None, 0, []

    plot_path = make_summary_file_path(binary_path, ".png")
    ensure_parent_directory(plot_path)

    sample_count = min(TEMPLATE_PLOT_SAMPLE_COUNT, int(number_of_templates))
    rng = np.random.default_rng(TEMPLATE_PLOT_SEED)
    sampled_indices = sorted(
        int(i)
        for i in rng.choice(
            int(number_of_templates), size=sample_count, replace=False
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        sample_count, 1, figsize=(14, max(2.2 * sample_count, 3.0)), dpi=150,
        sharex=True,
    )
    if sample_count == 1:
        axes = [axes]

    title_location = location if location else "--"
    fig.suptitle(
        f"{network}.{station}.{title_location}.{channel} "
        f"— {sample_count} of {number_of_templates} query templates "
        f"(length {template_length} samples)"
    )

    for ax, template_index in zip(axes, sampled_indices):
        start = int(template_index) * int(template_length)
        end = start + int(template_length)
        segment = np.asarray(
            all_template_data[start:end], dtype=np.float32
        )
        ax.plot(
            np.arange(segment.size), segment, linewidth=0.6, color="#1f77b4"
        )
        ax.set_ylabel(f"query #{template_index}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Sample index within template")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plot_path)
    plt.close(fig)
    return str(plot_path), sample_count, sampled_indices


def write_template_generation_report(
    binary_path,
    templates_csv,
    network,
    station,
    location,
    channel,
    number_of_templates,
    template_length,
    min_template_len,
    max_template_len,
    all_template_data,
    group_metadata,
    total_rows,
    selected_rows,
    loaded_templates,
):
    if not ENABLE_TEMPLATE_GENERATION_REPORT:
        return

    binary_path = Path(binary_path)
    json_path = make_summary_file_path(binary_path, ".json")
    group_report = finalize_template_group_metadata(group_metadata)
    sampling_rates = group_report.get("sampling_rates_hz", [])
    original_sampling_rates = group_report.get("original_sampling_rates_hz", [])
    downsample_factor_counts = group_report.get("downsample_factor_counts", [])
    applied_downsample_factors = [
        int(item["factor"]) for item in downsample_factor_counts
    ]

    plot_path = None
    plot_error = None
    plot_sample_count = 0
    plot_sample_indices = []
    try:
        plot_path, plot_sample_count, plot_sample_indices = write_template_sample_plot(
            binary_path,
            all_template_data,
            number_of_templates,
            template_length,
            network,
            station,
            location,
            channel,
        )
    except Exception as exc:
        plot_error = str(exc)
        print(
            f"Warning: Failed to write template sample plot for {binary_path}: {exc}"
        )

    metadata = {
        "dataset": "maule",
        "kind": "template_queries",
        "pipeline_version": PIPELINE_VERSION,
        "query_policy": "complete observed windows; intended duration rounded down to a multiple of 8",
        "created_at_utc": current_utc_timestamp(),
        "binary_file": str(binary_path),
        "binary_file_name": binary_path.name,
        "binary_file_size_bytes": binary_path.stat().st_size,
        "dtype": "float32",
        "network": network,
        "station": station,
        "location": location,
        "channel": channel,
        "number_of_templates": int(number_of_templates),
        "number_of_queries": int(number_of_templates),
        "template_length_points": int(template_length),
        "min_original_template_length_points": int(min_template_len),
        "max_original_template_length_points": int(max_template_len),
        "packed_total_points": int(all_template_data.size),
        "packed_layout": (
            "flat concatenation; template i occupies "
            "[i * template_length_points, (i + 1) * template_length_points)"
        ),
        "frequency": {
            "original_sampling_rates_hz": original_sampling_rates,
            "sampling_rates_hz": sampling_rates,
            "sampling_rate_hz": sampling_rates[0] if len(sampling_rates) == 1 else None,
            "bandpass_min_hz": FMIN,
            "bandpass_max_hz": FMAX,
            "downsample_factor_requested": DOWNSAMPLE_FACTOR,
            "downsample_factors_applied": applied_downsample_factors,
            "downsample_factor": (
                applied_downsample_factors[0]
                if len(applied_downsample_factors) == 1
                else None
            ),
        },
        "signal_processing": {
            "gap_fill": {
                "method": "Gaussian noise in masked gaps",
                "noise_scale": 3.0,
                "seed": 0,
            },
            "detrend": ["constant", "linear"],
            "taper": {"max_percentage": 0.02, "type": "cosine"},
            "filter": {
                "type": "bandpass",
                "freqmin": FMIN,
                "freqmax": FMAX,
                "zerophase": True,
            },
            "downsample": {
                "requested_factor": DOWNSAMPLE_FACTOR,
                "applied_factor_counts": downsample_factor_counts,
                "method": (
                    "ObsPy Trace.decimate(no_filter=True) after bandpass when "
                    "freqmax is below the target Nyquist; otherwise the trace "
                    "remains at its original sampling rate"
                ),
            },
            "trim": "actual available bounds for each loaded template window",
        },
        "source": {
            "sds_root": SDS_ROOT,
            "templates_csv": templates_csv,
            "csv_total_rows_read": int(total_rows),
            "csv_candidate_rows_selected": int(selected_rows),
            "loaded_matching_templates": int(loaded_templates),
            "group": group_report,
        },
        "selection_filters": {
            "networks": selected_values(SELECT_NETWORKS),
            "stations": selected_values(SELECT_STATIONS),
            "locations": selected_values(SELECT_LOCATIONS),
            "channels": selected_values(SELECT_CHANNELS),
        },
        "amplitude_stats": array_stats(all_template_data),
        "summary_output": {
            "json_path": str(json_path),
            "plot_path": plot_path,
            "plot_error": plot_error,
            "plot_sample_count": int(plot_sample_count),
            "plot_sample_indices": list(plot_sample_indices),
            "plot_sample_target": TEMPLATE_PLOT_SAMPLE_COUNT,
            "plot_sample_seed": TEMPLATE_PLOT_SEED,
            "plot_style": (
                "randomly sampled query templates rendered as vertically "
                "stacked subplots (one axis per sampled template)"
            ),
        },
    }
    write_json_file(json_path, metadata)
    print(f">> Summary: Wrote template metadata report to {json_path}")


def main():
    templates_csv = (
        "/path/to/raw/maule/"
        "templates_all_stations_channels.csv"
    )

    template_map = {}
    template_group_metadata = {}
    total_rows = 0
    selected_rows = 0
    loaded_templates = 0
    rejected_templates = 0
    rejected_duration_mismatches = 0

    csv_reader = pd.read_csv(
        templates_csv,
        usecols=["station", "channel", "starttime", "endtime"],
        chunksize=TEMPLATE_CHUNKSIZE,
    )

    with tqdm(desc="CSV rows", unit="row") as csv_progress, tqdm(
        desc="Template candidates", unit="template"
    ) as candidate_progress:
        for chunk in csv_reader:
            total_rows += len(chunk)
            # Keep every channel of a station in the same array task.
            import zlib
            stride = int(os.getenv("MAULE_TEMPLATE_TASK_COUNT", "1"))
            offset = int(os.getenv("MAULE_TEMPLATE_TASK_ID", "0"))
            if stride > 1:
                chunk = chunk[chunk["station"].map(
                    lambda station: zlib.crc32(str(station).encode()) % stride == offset)]
            csv_progress.update(len(chunk))
            if SELECT_STATIONS is not None:
                chunk = chunk[chunk["station"].isin(SELECT_STATIONS)]
            if SELECT_CHANNELS is not None:
                chunk = chunk[chunk["channel"].isin(SELECT_CHANNELS)]
            if chunk.empty:
                continue
            selected_rows += len(chunk)

            for row in chunk.itertuples(index=True):
                idx = row.Index
                requested_station = row.station
                requested_channel = row.channel
                requested_start = UTCDateTime(row.starttime)
                requested_end = UTCDateTime(row.endtime)

                if (
                    abs(
                        (requested_end - requested_start)
                        - NOMINAL_QUERY_DURATION_SECONDS
                    )
                    > QUERY_DURATION_TOLERANCE_SECONDS
                ):
                    rejected_duration_mismatches += 1
                    continue

                candidate_progress.set_postfix_str(
                    f"{requested_station}.{requested_channel}"
                )

                st, stm, stf, _, _ = load_and_prepare_trace(
                    SDS_ROOT,
                    requested_start,
                    requested_end,
                    "*",
                    requested_station,
                    "*",
                    requested_channel,
                )
                candidate_progress.update(1)

                if len(st) == 0 or len(stf) == 0:
                    tqdm.write(
                        f"Skipping [{idx}] *. {requested_station} .* .{requested_channel} "
                        "due to empty traces."
                    )
                    continue

                assert len(stf) == 1, f"[{idx}] Expected 1 trace, got {len(stf)}"

                template_trace = stf[0]
                network, station, location, channel = (
                    template_trace.stats.network,
                    template_trace.stats.station,
                    template_trace.stats.location,
                    template_trace.stats.channel,
                )

                if not matches_selection(network, station, location, channel):
                    continue

                if loaded_templates < TEMPLATE_LOG_LIMIT:
                    tqdm.write(
                        f"[{idx}] {network}.{station}.{location}.{channel} | "
                        f"SR: {template_trace.stats.sampling_rate} | "
                        f"L: {len(template_trace.data)} | "
                        f"F3p: {template_trace.data[0:3]} | "
                        f"L3p: {template_trace.data[-3:]}"
                    )
                elif loaded_templates == TEMPLATE_LOG_LIMIT:
                    tqdm.write(
                        "Template log limit reached; suppressing per-template trace output."
                    )

                data = complete_template(template_trace, stm, requested_start, requested_end)
                if data is None:
                    rejected_templates += 1
                    continue

                key = (network, station, location, channel)
                template_map.setdefault(key, []).append(data)
                if ENABLE_TEMPLATE_GENERATION_REPORT:
                    update_template_group_metadata(
                        template_group_metadata,
                        key,
                        idx,
                        requested_start,
                        requested_end,
                        template_trace,
                    )
                loaded_templates += 1

    print(
        f"Read {total_rows} CSV rows, selected {selected_rows} candidate rows, "
        f"loaded {loaded_templates} matching templates; rejected {rejected_templates} incomplete/gapped/constant windows; "
        f"rejected {rejected_duration_mismatches} rows with a requested duration outside "
        f"{NOMINAL_QUERY_DURATION_SECONDS}s +/- {QUERY_DURATION_TOLERANCE_SECONDS}s."
    )

    print("\n\n\n")
    for key, templates in tqdm(
        template_map.items(), total=len(template_map), desc="Template groups", unit="group"
    ):
        print(f"{key} -> {len(templates)}")
    print("\n\n\n")

    for key, templates in template_map.items():
        network, station, location, channel = key
        min_template_len = min(len(template) for template in templates)
        max_template_len = max(len(template) for template in templates)
        if min_template_len != max_template_len:
            tqdm.write(
                f"{key}: template lengths range {min_template_len}-{max_template_len} "
                "points (sample-grid rounding); truncating all to the shortest."
            )
        all_template_len = pad_to_mul(min_template_len, mul=8)

        print(
            f"Processing {len(templates)} templates for "
            f"{network}.{station}.{location}.{channel} with length {all_template_len}, "
            f"MaxLen: {max_template_len}, MinLen: {min_template_len}"
        )

        all_template_data = np.empty(
            len(templates) * all_template_len, dtype=np.float32
        )
        for idx, template in enumerate(
            tqdm(
                templates,
                desc=f"Packing {network}.{station}.{location}.{channel}",
                unit="template",
            )
        ):
            assert len(template) >= all_template_len, (
                f"Template {idx} has smaller length: "
                f"{len(template)} != {all_template_len}"
            )

            data = np.asarray(template[0:all_template_len], dtype=np.float32)
            assert not np.isnan(data).any()

            start = idx * all_template_len
            all_template_data[start : start + all_template_len] = data

        filename = str(
            Path(DIRECTORY)
            / (
                f"tem{len(templates)}.{network}_{station}_{location}_{channel}"
                f".len{all_template_len}.fmin{FMIN}.fmax{FMAX}.bin"
            )
        )
        save_binary_file(filename, all_template_data)
        write_template_generation_report(
            filename,
            templates_csv,
            network,
            station,
            location,
            channel,
            len(templates),
            all_template_len,
            min_template_len,
            max_template_len,
            all_template_data,
            template_group_metadata.get(key),
            total_rows,
            selected_rows,
            loaded_templates,
        )


if __name__ == "__main__":
    main()
    
