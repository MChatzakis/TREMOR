from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality import PIPELINE_VERSION, complete_template
import obspy
import pandas as pd
from obspy import UTCDateTime

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
    current_utc_timestamp,
    ensure_parent_directory,
    env_flag,
    load_and_prepare_trace,
    matches_selection,
    selected_values,
    tqdm,
    utc_datetime_to_string,
)

DIRECTORY = os.getenv(
    "SEIFR_TEMPLATE_OUTPUT_DIR",
    "/path/to/data/seifr/templates/",
)
TEMPLATES_CSV = os.getenv("SEIFR_TEMPLATE_CSV")
EVENTS_QML = os.getenv(
    "SEIFR_EVENTS_QML",
    "/path/to/raw/seifr/event.qml",
)
TEMPLATE_SOURCE = os.getenv("SEIFR_TEMPLATE_SOURCE", "auto").strip().lower()
TEMPLATE_WINDOW_LENGTH_SEC = float(os.getenv("SEIFR_TEMPLATE_WINDOW_LENGTH_SEC", "120"))
TEMPLATE_LEAD_TIME_SEC = float(os.getenv("SEIFR_TEMPLATE_LEAD_TIME_SEC", "2"))

ENABLE_TEMPLATE_GENERATION_REPORT = env_flag("SEIFR_ENABLE_TEMPLATE_REPORT", True)
SUMMARY_OUTPUT_DIR = Path(
    os.getenv(
        "SEIFR_SUMMARY_OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "metadata"),
    )
)
TEMPLATE_REPORT_ROW_PREVIEW_LIMIT = int(
    os.getenv("SEIFR_TEMPLATE_REPORT_ROW_PREVIEW_LIMIT", "20")
)
TEMPLATE_PLOT_SAMPLE_COUNT = int(os.getenv("SEIFR_TEMPLATE_PLOT_SAMPLE_COUNT", "5"))
TEMPLATE_PLOT_SEED = int(os.getenv("SEIFR_TEMPLATE_PLOT_SEED", "0"))


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


def write_json_file(filename, data):
    ensure_parent_directory(filename)
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)


def selected_template_source():
    if TEMPLATE_SOURCE not in {"auto", "csv", "qml"}:
        raise ValueError(
            "SEIFR_TEMPLATE_SOURCE must be one of: auto, csv, qml "
            f"(got {TEMPLATE_SOURCE!r})"
        )

    if TEMPLATE_SOURCE == "csv":
        if not TEMPLATES_CSV:
            raise ValueError("SEIFR_TEMPLATE_CSV is required when source=csv")
        return "csv", Path(TEMPLATES_CSV)

    if TEMPLATE_SOURCE == "qml":
        return "qml", Path(EVENTS_QML)

    if TEMPLATES_CSV:
        return "csv", Path(TEMPLATES_CSV)
    return "qml", Path(EVENTS_QML)


def validate_configuration():
    mode, source_path = selected_template_source()
    if not source_path.is_file():
        env_name = "SEIFR_TEMPLATE_CSV" if mode == "csv" else "SEIFR_EVENTS_QML"
        raise FileNotFoundError(f"{env_name} not found: {source_path}")
    if TEMPLATE_WINDOW_LENGTH_SEC <= 0:
        raise ValueError("SEIFR_TEMPLATE_WINDOW_LENGTH_SEC must be > 0")
    if TEMPLATE_LEAD_TIME_SEC < 0:
        raise ValueError("SEIFR_TEMPLATE_LEAD_TIME_SEC must be >= 0")
    return mode, source_path


def make_template_group_metadata():
    return {
        "template_count": 0,
        "source_candidate_indices_preview": [],
        "event_public_ids_preview": [],
        "pick_public_ids_preview": [],
        "phase_hint_counts": Counter(),
        "requested_start_min": None,
        "requested_end_max": None,
        "trace_start_min": None,
        "trace_end_max": None,
        "sampling_rates_hz": set(),
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


def preview_value(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def append_preview(values, value):
    if value is None:
        return
    if len(values) < TEMPLATE_REPORT_ROW_PREVIEW_LIMIT:
        values.append(preview_value(value))


def update_template_group_metadata(
    template_group_metadata,
    key,
    source_candidate_index,
    requested_start,
    requested_end,
    template_trace,
    phase_hint=None,
    event_public_id=None,
    pick_public_id=None,
):
    group = template_group_metadata.setdefault(key, make_template_group_metadata())
    group["template_count"] += 1
    append_preview(group["source_candidate_indices_preview"], source_candidate_index)
    append_preview(group["event_public_ids_preview"], event_public_id)
    append_preview(group["pick_public_ids_preview"], pick_public_id)

    if phase_hint:
        group["phase_hint_counts"][str(phase_hint)] += 1

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
    group["sampling_rates_hz"].add(float(template_trace.stats.sampling_rate))
    group["original_length_counts"][int(len(template_trace.data))] += 1


def finalize_template_group_metadata(group):
    if not group:
        return {}

    return {
        "template_count": group["template_count"],
        "source_candidate_indices_preview": group["source_candidate_indices_preview"],
        "source_candidate_indices_preview_limit": TEMPLATE_REPORT_ROW_PREVIEW_LIMIT,
        "event_public_ids_preview": group["event_public_ids_preview"],
        "pick_public_ids_preview": group["pick_public_ids_preview"],
        "phase_hint_counts": [
            {"phase": phase, "count": count}
            for phase, count in sorted(group["phase_hint_counts"].items())
        ],
        "requested_start_min": utc_datetime_to_string(group["requested_start_min"]),
        "requested_end_max": utc_datetime_to_string(group["requested_end_max"]),
        "trace_start_min": utc_datetime_to_string(group["trace_start_min"]),
        "trace_end_max": utc_datetime_to_string(group["trace_end_max"]),
        "sampling_rates_hz": sorted(group["sampling_rates_hz"]),
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
        sample_count,
        1,
        figsize=(14, max(2.2 * sample_count, 3.0)),
        dpi=150,
        sharex=True,
    )
    if sample_count == 1:
        axes = [axes]

    title_location = location if location else "--"
    fig.suptitle(
        f"{network}.{station}.{title_location}.{channel} - "
        f"{sample_count} of {number_of_templates} query templates "
        f"(length {template_length} samples)"
    )

    for ax, template_index in zip(axes, sampled_indices):
        start = int(template_index) * int(template_length)
        end = start + int(template_length)
        segment = np.asarray(all_template_data[start:end], dtype=np.float32)
        ax.plot(np.arange(segment.size), segment, linewidth=0.6, color="#1f77b4")
        ax.set_ylabel(f"query #{template_index}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Sample index within template")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plot_path)
    plt.close(fig)
    return str(plot_path), sample_count, sampled_indices


def write_template_generation_report(
    binary_path,
    source_report,
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
):
    if not ENABLE_TEMPLATE_GENERATION_REPORT:
        return

    binary_path = Path(binary_path)
    json_path = make_summary_file_path(binary_path, ".json")
    group_report = finalize_template_group_metadata(group_metadata)
    sampling_rates = group_report.get("sampling_rates_hz", [])

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
        print(f"Warning: Failed to write template sample plot for {binary_path}: {exc}")

    metadata = {
        "dataset": "seiFR",
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
            "sampling_rates_hz": sampling_rates,
            "sampling_rate_hz": sampling_rates[0] if len(sampling_rates) == 1 else None,
            "bandpass_min_hz": FMIN,
            "bandpass_max_hz": FMAX,
            "downsample_factor": DOWNSAMPLE_FACTOR,
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
                "factor": DOWNSAMPLE_FACTOR,
                "method": "ObsPy decimate(no_filter=True) after anti-alias bandpass",
            },
            "trim": "actual available bounds for each loaded template window",
        },
        "source": {
            **source_report,
            "sds_root": SDS_ROOT,
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


def clean_code(value, default=""):
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value)
    if text.lower() == "nan":
        return default
    return text.strip()


def csv_code(row, column, default):
    if column not in row:
        return default
    return clean_code(row[column], default)


def get_pick_info(pick):
    waveform_id = pick.waveform_id
    network = clean_code(getattr(waveform_id, "network_code", None), "*")
    station = clean_code(getattr(waveform_id, "station_code", None), "*")
    location = clean_code(getattr(waveform_id, "location_code", None), "")
    channel = clean_code(getattr(waveform_id, "channel_code", None), "*")

    if not station or station == "*":
        parts = str(waveform_id.id).split(".")
        if len(parts) >= 4:
            network = parts[0] or "*"
            station = parts[1] or "*"
            location = parts[2]
            channel = parts[3] or "*"

    if channel != "*" and len(channel) < 3:
        channel = "*" + channel

    return network, station, location, channel


def matches_requested_selection(network, station, location, channel):
    return (
        (SELECT_NETWORKS is None or network == "*" or network in SELECT_NETWORKS)
        and (SELECT_STATIONS is None or station == "*" or station in SELECT_STATIONS)
        and (SELECT_LOCATIONS is None or location == "*" or location in SELECT_LOCATIONS)
        and (SELECT_CHANNELS is None or channel == "*" or channel in SELECT_CHANNELS)
    )


def filter_csv_chunk(chunk):
    missing_columns = [
        column
        for column in ("station", "channel", "starttime", "endtime")
        if column not in chunk.columns
    ]
    if missing_columns:
        raise ValueError(
            "Template CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    filtered = chunk
    if SELECT_NETWORKS is not None and "network" in filtered.columns:
        filtered = filtered[
            filtered["network"].fillna("").astype(str).str.strip().isin(SELECT_NETWORKS)
        ]
    if SELECT_STATIONS is not None:
        filtered = filtered[
            filtered["station"].fillna("").astype(str).str.strip().isin(SELECT_STATIONS)
        ]
    if SELECT_LOCATIONS is not None and "location" in filtered.columns:
        filtered = filtered[
            filtered["location"].fillna("").astype(str).str.strip().isin(SELECT_LOCATIONS)
        ]
    if SELECT_CHANNELS is not None:
        filtered = filtered[
            filtered["channel"].fillna("").astype(str).str.strip().isin(SELECT_CHANNELS)
        ]

    return filtered


def load_and_store_template(
    template_map,
    template_group_metadata,
    source_candidate_index,
    requested_start,
    requested_end,
    network,
    station,
    location,
    channel,
    loaded_templates,
    phase_hint=None,
    event_public_id=None,
    pick_public_id=None,
):
    st, stm, stf, _, _ = load_and_prepare_trace(
        SDS_ROOT,
        requested_start,
        requested_end,
        network,
        station,
        location,
        channel,
    )

    if len(st) == 0 or len(stf) == 0:
        tqdm.write(
            f"Skipping [{source_candidate_index}] {network}.{station}."
            f"{location}.{channel} due to empty traces."
        )
        return "empty", loaded_templates

    if len(stf) != 1:
        raise ValueError(f"[{source_candidate_index}] Expected 1 trace, got {len(stf)}")

    template_trace = stf[0]
    network = template_trace.stats.network
    station = template_trace.stats.station
    location = template_trace.stats.location
    channel = template_trace.stats.channel

    if not matches_selection(network, station, location, channel):
        return "selection", loaded_templates

    if loaded_templates < TEMPLATE_LOG_LIMIT:
        tqdm.write(
            f"[{source_candidate_index}] {network}.{station}.{location}.{channel} | "
            f"SR: {template_trace.stats.sampling_rate} | "
            f"L: {len(template_trace.data)} | "
            f"F3p: {template_trace.data[0:3]} | "
            f"L3p: {template_trace.data[-3:]}"
        )
    elif loaded_templates == TEMPLATE_LOG_LIMIT:
        tqdm.write("Template log limit reached; suppressing per-template trace output.")

    data = complete_template(template_trace, stm, requested_start, requested_end)
    if data is None:
        return "incomplete", loaded_templates

    key = (network, station, location, channel)
    template_map.setdefault(key, []).append(data)
    if ENABLE_TEMPLATE_GENERATION_REPORT:
        update_template_group_metadata(
            template_group_metadata,
            key,
            source_candidate_index,
            requested_start,
            requested_end,
            template_trace,
            phase_hint=phase_hint,
            event_public_id=event_public_id,
            pick_public_id=pick_public_id,
        )

    return "loaded", loaded_templates + 1


def collect_templates_from_csv(source_path):
    template_map = {}
    template_group_metadata = {}
    stats = {
        "template_source": "csv",
        "templates_csv": str(source_path),
        "csv_total_rows_read": 0,
        "csv_candidate_rows_selected": 0,
        "loaded_matching_templates": 0,
        "skipped_empty_traces": 0,
        "skipped_incomplete_traces": 0,
        "skipped_by_selection_after_load": 0,
    }

    csv_reader = pd.read_csv(source_path, chunksize=TEMPLATE_CHUNKSIZE)

    with tqdm(desc="CSV rows", unit="row") as csv_progress, tqdm(
        desc="Template candidates", unit="template"
    ) as candidate_progress:
        for chunk in csv_reader:
            stats["csv_total_rows_read"] += len(chunk)
            csv_progress.update(len(chunk))
            chunk = filter_csv_chunk(chunk)
            if chunk.empty:
                continue
            stats["csv_candidate_rows_selected"] += len(chunk)

            for idx, row in chunk.iterrows():
                network = csv_code(row, "network", "*")
                station = csv_code(row, "station", "*")
                location = csv_code(row, "location", "*")
                channel = csv_code(row, "channel", "*")
                requested_start = UTCDateTime(row["starttime"])
                requested_end = UTCDateTime(row["endtime"])
                candidate_progress.set_postfix_str(f"{station}.{channel}")

                status, loaded_templates = load_and_store_template(
                    template_map,
                    template_group_metadata,
                    idx,
                    requested_start,
                    requested_end,
                    network,
                    station,
                    location,
                    channel,
                    stats["loaded_matching_templates"],
                )
                candidate_progress.update(1)
                stats["loaded_matching_templates"] = loaded_templates
                if status == "incomplete":
                    stats["skipped_incomplete_traces"] += 1
                elif status == "empty":
                    stats["skipped_empty_traces"] += 1
                elif status == "selection":
                    stats["skipped_by_selection_after_load"] += 1

    return template_map, template_group_metadata, stats


def collect_templates_from_qml(source_path):
    catalog = obspy.read_events(str(source_path))
    total_arrivals = sum(len(origin.arrivals) for event in catalog for origin in event.origins)

    template_map = {}
    template_group_metadata = {}
    stats = {
        "template_source": "qml",
        "events_qml": str(source_path),
        "catalog_event_count": len(catalog),
        "catalog_arrival_count": total_arrivals,
        "catalog_arrivals_selected": 0,
        "loaded_matching_templates": 0,
        "skipped_missing_pick": 0,
        "skipped_by_selection_before_load": 0,
        "skipped_by_selection_after_load": 0,
        "skipped_empty_traces": 0,
        "skipped_incomplete_traces": 0,
        "template_window_length_seconds": TEMPLATE_WINDOW_LENGTH_SEC,
        "template_lead_time_seconds": TEMPLATE_LEAD_TIME_SEC,
    }

    source_candidate_index = 0
    with tqdm(total=total_arrivals, desc="Catalog arrivals", unit="arrival") as progress:
        for event in catalog:
            event_public_id = getattr(event, "resource_id", None)
            event_public_id = str(event_public_id) if event_public_id else None
            for origin in event.origins:
                for arrival in origin.arrivals:
                    source_candidate_index += 1
                    progress.update(1)
                    pick = arrival.pick_id.get_referred_object()
                    if pick is None:
                        stats["skipped_missing_pick"] += 1
                        continue

                    network, station, location, channel = get_pick_info(pick)
                    phase_hint = clean_code(getattr(pick, "phase_hint", None), None)
                    pick_public_id = getattr(pick, "resource_id", None)
                    pick_public_id = str(pick_public_id) if pick_public_id else None

                    if not matches_requested_selection(network, station, location, channel):
                        stats["skipped_by_selection_before_load"] += 1
                        continue

                    requested_start = pick.time - TEMPLATE_LEAD_TIME_SEC
                    requested_end = requested_start + TEMPLATE_WINDOW_LENGTH_SEC
                    progress.set_postfix_str(f"{station}.{channel}")
                    stats["catalog_arrivals_selected"] += 1

                    status, loaded_templates = load_and_store_template(
                        template_map,
                        template_group_metadata,
                        source_candidate_index,
                        requested_start,
                        requested_end,
                        network,
                        station,
                        location,
                        channel,
                        stats["loaded_matching_templates"],
                        phase_hint=phase_hint,
                        event_public_id=event_public_id,
                        pick_public_id=pick_public_id,
                    )
                    stats["loaded_matching_templates"] = loaded_templates
                    if status == "incomplete":
                        stats["skipped_incomplete_traces"] += 1
                    elif status == "empty":
                        stats["skipped_empty_traces"] += 1
                    elif status == "selection":
                        stats["skipped_by_selection_after_load"] += 1

    return template_map, template_group_metadata, stats


def print_source_summary(stats):
    if stats["template_source"] == "csv":
        print(
            f"Read {stats['csv_total_rows_read']} CSV rows, selected "
            f"{stats['csv_candidate_rows_selected']} candidate rows, loaded "
            f"{stats['loaded_matching_templates']} matching templates."
        )
        return

    print(
        f"Read {stats['catalog_event_count']} QML events with "
        f"{stats['catalog_arrival_count']} arrivals, selected "
        f"{stats['catalog_arrivals_selected']} arrivals, loaded "
        f"{stats['loaded_matching_templates']} matching templates."
    )


def write_template_groups(template_map, template_group_metadata, source_report):
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
        template_length = pad_to_mul(min_template_len, mul=8)

        if template_length <= 0:
            print(f"Skipping {network}.{station}.{location}.{channel}: zero length")
            continue

        print(
            f"Processing {len(templates)} templates for "
            f"{network}.{station}.{location}.{channel} with length {template_length}, "
            f"MaxLen: {max_template_len}, MinLen: {min_template_len}"
        )

        all_template_data = np.empty(len(templates) * template_length, dtype=np.float32)
        for idx, template in enumerate(
            tqdm(
                templates,
                desc=f"Packing {network}.{station}.{location}.{channel}",
                unit="template",
            )
        ):
            if len(template) < template_length:
                raise ValueError(
                    f"Template {idx} has smaller length: "
                    f"{len(template)} < {template_length}"
                )

            data = np.asarray(template[:template_length], dtype=np.float32)
            if np.isnan(data).any():
                raise ValueError(
                    f"Template {idx} for {network}.{station}.{location}.{channel} "
                    "contains NaN values"
                )

            start = idx * template_length
            all_template_data[start : start + template_length] = data

        filename = str(
            Path(DIRECTORY)
            / (
                f"tem{len(templates)}.{network}_{station}_{location}_{channel}"
                f".len{template_length}.fmin{FMIN}.fmax{FMAX}.bin"
            )
        )
        save_binary_file(filename, all_template_data)
        write_template_generation_report(
            filename,
            source_report,
            network,
            station,
            location,
            channel,
            len(templates),
            template_length,
            min_template_len,
            max_template_len,
            all_template_data,
            template_group_metadata.get(key),
        )


def main():
    mode, source_path = validate_configuration()

    print("Active template generation filters:")
    print(f"  networks: {SELECT_NETWORKS}")
    print(f"  stations: {SELECT_STATIONS}")
    print(f"  locations: {SELECT_LOCATIONS}")
    print(f"  channels: {SELECT_CHANNELS}")
    print(f"Template source: {mode} ({source_path})")
    print(f"Template output: {DIRECTORY}")
    print(f"Template report enabled: {ENABLE_TEMPLATE_GENERATION_REPORT}")

    if mode == "csv":
        template_map, template_group_metadata, source_report = collect_templates_from_csv(
            source_path
        )
    else:
        template_map, template_group_metadata, source_report = collect_templates_from_qml(
            source_path
        )

    print_source_summary(source_report)

    if not template_map:
        print("No template groups selected. Nothing to process.")
        return

    write_template_groups(template_map, template_group_metadata, source_report)


if __name__ == "__main__":
    main()
