from obspy import UTCDateTime
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality import PIPELINE_VERSION, chunk_samples, complete_template

from shared_pipeline import (
    CHANNEL_PREFIXES,
    DOWNSAMPLE_FACTOR,
    ENABLE_INVENTORY_REPORT,
    FMAX,
    FMIN,
    GENERATION_ENTRY_OFFSET,
    GENERATION_ENTRY_STRIDE,
    MAX_GENERATION_ENTRIES,
    SDS_ROOT,
    SELECT_CHANNELS,
    SELECT_LOCATIONS,
    SELECT_NETWORKS,
    SELECT_STATIONS,
    ensure_parent_directory,
    filter_inventory,
    get_trace_downsample_factor,
    get_trace_original_sampling_rate,
    load_and_fill_sds_traces,
    load_and_prepare_trace,
    matches_selected_channel,
    matches_selected_location,
    matches_selected_network,
    matches_selected_station,
    selection_filters_enabled,
    tqdm,
)

SECONDS_PER_DAY = 86400
DIRECTORY = os.getenv(
    "MAULE_WAVEFORM_OUTPUT_DIR",
    "/path/to/data/maule/waveforms/",
)
GENERATION_CHUNK_DAYS = int(os.getenv("MAULE_GENERATION_CHUNK_DAYS", "1"))
_generation_max_days = os.getenv("MAULE_GENERATION_MAX_DAYS")
GENERATION_MAX_DAYS = (
    int(_generation_max_days) if _generation_max_days is not None else None
)

_FALSE_ENV_VALUES = {"0", "false", "no", "off", ""}


def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


# Toggle the per-binary JSON metadata and waveform overview PNG sidecars.
ENABLE_GENERATION_REPORT = env_flag("MAULE_ENABLE_WAVEFORM_REPORT", True)
SUMMARY_OUTPUT_DIR = Path(
    os.getenv(
        "MAULE_SUMMARY_OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "metadata"),
    )
)
WAVEFORM_PLOT_MAX_POINTS = int(os.getenv("MAULE_WAVEFORM_PLOT_MAX_POINTS", "200000"))
WAVEFORM_PLOT_POINTS_PER_CHUNK = int(
    os.getenv("MAULE_WAVEFORM_PLOT_POINTS_PER_CHUNK", "4000")
)


def save_binary_file(filename, data):
    ensure_parent_directory(filename)
    data = np.asarray(data, dtype=np.float32)
    print(f">> BinFile: Writing {data.size} points to {filename}")
    data.tofile(filename)


def append_binary_chunk(file_handle, data):
    data = np.asarray(data, dtype=np.float32)
    data.tofile(file_handle)
    return data.size


def make_waveform_filename(output_dir, network, station, location, channel, npts):
    return (
        str(Path(output_dir) / f"{network}_{station}_{location}_{channel}.len{npts}.bin")
    )


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


def make_data_stats():
    return {
        "finite_points": 0,
        "nan_points": 0,
        "inf_points": 0,
        "min": None,
        "max": None,
        "sum": 0.0,
    }


def update_data_stats(stats, data):
    if data.size == 0:
        return

    data = np.asarray(data)
    finite_mask = np.isfinite(data)
    finite_points = int(finite_mask.sum())
    stats["finite_points"] += finite_points
    stats["nan_points"] += int(np.isnan(data).sum())
    stats["inf_points"] += int(np.isinf(data).sum())

    if finite_points == 0:
        return

    finite_data = data if finite_points == data.size else data[finite_mask]
    chunk_min = float(finite_data.min())
    chunk_max = float(finite_data.max())
    stats["min"] = chunk_min if stats["min"] is None else min(stats["min"], chunk_min)
    stats["max"] = chunk_max if stats["max"] is None else max(stats["max"], chunk_max)
    stats["sum"] += float(finite_data.sum(dtype=np.float64))


def finalize_data_stats(stats):
    finite_points = stats["finite_points"]
    mean = stats["sum"] / finite_points if finite_points else None
    return {
        "finite_points": finite_points,
        "nan_points": stats["nan_points"],
        "inf_points": stats["inf_points"],
        "min": stats["min"],
        "max": stats["max"],
        "mean": mean,
    }


def make_plot_preview():
    return {"x_parts": [], "y_parts": [], "points": 0}


def minmax_preview(data, start_index, max_points):
    data = np.asarray(data, dtype=np.float32)
    npts = data.size
    if npts == 0 or max_points <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)

    if npts <= max_points:
        x = start_index + np.arange(npts, dtype=np.float64)
        return x, data

    bin_count = max(1, max_points // 2)
    bin_size = max(1, npts // bin_count)
    trimmed = (npts // bin_size) * bin_size
    reshaped = data[:trimmed].reshape(-1, bin_size)
    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)
    centers = start_index + (
        np.arange(reshaped.shape[0], dtype=np.float64) * bin_size
    ) + (0.5 * bin_size)

    x = np.repeat(centers, 2)
    y = np.empty(x.size, dtype=np.float32)
    y[0::2] = mins
    y[1::2] = maxs

    if trimmed < npts:
        x = np.concatenate([x, np.array([start_index + npts - 1], dtype=np.float64)])
        y = np.concatenate([y, np.array([data[-1]], dtype=np.float32)])

    return x, y


def compress_plot_preview(preview):
    if preview["points"] <= WAVEFORM_PLOT_MAX_POINTS:
        return

    x = np.concatenate(preview["x_parts"])
    y = np.concatenate(preview["y_parts"])
    if y.size <= WAVEFORM_PLOT_MAX_POINTS:
        preview["x_parts"] = [x]
        preview["y_parts"] = [y]
        preview["points"] = y.size
        return

    indices = np.linspace(0, y.size - 1, WAVEFORM_PLOT_MAX_POINTS, dtype=np.int64)
    preview["x_parts"] = [x[indices]]
    preview["y_parts"] = [y[indices]]
    preview["points"] = WAVEFORM_PLOT_MAX_POINTS


def append_plot_preview(preview, data, start_index):
    x, y = minmax_preview(data, start_index, WAVEFORM_PLOT_POINTS_PER_CHUNK)
    if y.size == 0:
        return
    preview["x_parts"].append(x)
    preview["y_parts"].append(y)
    preview["points"] += y.size
    if preview["points"] > 2 * WAVEFORM_PLOT_MAX_POINTS:
        compress_plot_preview(preview)


def plot_preview_arrays(preview):
    if not preview["y_parts"]:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)
    compress_plot_preview(preview)
    return np.concatenate(preview["x_parts"]), np.concatenate(preview["y_parts"])


def write_json_file(filename, data):
    ensure_parent_directory(filename)
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)


def write_waveform_plot(filename, preview, metadata):
    x, y = plot_preview_arrays(preview)
    if y.size == 0:
        return None, 0

    ensure_parent_directory(filename)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title_location = metadata["location"] if metadata["location"] else "--"
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.plot(x, y, linewidth=0.35, color="#1f77b4")
    ax.set_title(
        f"{metadata['network']}.{metadata['station']}.{title_location}."
        f"{metadata['channel']} ({metadata['number_of_points']} samples)"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename, int(y.size)


def write_waveform_generation_report(binary_path, metadata, preview):
    if not ENABLE_GENERATION_REPORT:
        return

    json_path = make_summary_file_path(binary_path, ".json")
    plot_path = make_summary_file_path(binary_path, ".png")
    plot_error = None
    plotted_points = 0
    written_plot_path = None

    try:
        written_plot_path, plotted_points = write_waveform_plot(
            plot_path, preview, metadata
        )
    except Exception as exc:
        plot_error = str(exc)
        print(f"Warning: Failed to write waveform plot for {binary_path}: {exc}")

    report = dict(metadata)
    report["summary_output"] = {
        "json_path": str(json_path),
        "plot_path": str(written_plot_path) if written_plot_path else None,
        "plot_error": plot_error,
        "plot_points": plotted_points,
        "plot_max_points": WAVEFORM_PLOT_MAX_POINTS,
        "plot_points_per_chunk": WAVEFORM_PLOT_POINTS_PER_CHUNK,
        "plot_style": "streamed min/max overview over the full written waveform",
    }
    write_json_file(json_path, report)
    print(f">> Summary: Wrote metadata report to {json_path}")


def update_time_bounds(current_start, current_end, chunk_start, chunk_end):
    if chunk_start is not None:
        current_start = (
            chunk_start if current_start is None else min(current_start, chunk_start)
        )
    if chunk_end is not None:
        current_end = chunk_end if current_end is None else max(current_end, chunk_end)
    return current_start, current_end


def generate_waveform_binary(network, station, location, channel, t_start, t_end):
    output_dir = Path(DIRECTORY)
    ensure_parent_directory(output_dir / "placeholder")
    tmp_path = output_dir / f"{network}_{station}_{location}_{channel}.tmp.bin"
    total_points = 0
    chunk_seconds = GENERATION_CHUNK_DAYS * SECONDS_PER_DAY
    if GENERATION_MAX_DAYS is not None:
        t_end = min(t_end, t_start + GENERATION_MAX_DAYS * SECONDS_PER_DAY)
    chunk_start = t_start
    chunks_written = 0
    total_chunks = int(np.ceil((t_end - t_start) / chunk_seconds))
    empty_chunks = 0
    boundary_samples_dropped = 0
    previous_written_end = None
    chunk_map = []
    filled_gap_samples = 0
    gap_reference_samples = 0
    original_sampling_rates = set()
    sampling_rates = set()
    downsample_factors = set()
    deltas = set()
    actual_start = None
    actual_end = None
    data_stats = make_data_stats()
    plot_preview = make_plot_preview()

    with open(tmp_path, "wb") as raw_bin_file:
        with tqdm(
            total=total_chunks,
            desc=f"{network}.{station}.{location}.{channel}",
            unit="chunk",
        ) as progress:
            while chunk_start < t_end:
                chunk_end = min(chunk_start + chunk_seconds, t_end)
                progress.set_postfix_str(f"{chunk_start.date}->{chunk_end.date}")
                (
                    st,
                    stm,
                    stf,
                    chunk_actual_start,
                    chunk_actual_end,
                ) = load_and_prepare_trace(
                    SDS_ROOT,
                    chunk_start,
                    chunk_end,
                    network,
                    station,
                    location,
                    channel,
                )

                if len(st) == 0 or len(stf) == 0:
                    empty_chunks += 1
                    chunk_start = chunk_end
                    progress.update(1)
                    continue

                assert len(stf) == 1
                wf = stf[0]
                actual_start, actual_end = update_time_bounds(
                    actual_start, actual_end, chunk_actual_start, chunk_actual_end
                )
                original_sampling_rates.add(get_trace_original_sampling_rate(wf))
                sampling_rates.add(float(wf.stats.sampling_rate))
                downsample_factors.add(get_trace_downsample_factor(wf))
                deltas.add(float(wf.stats.delta))

                if len(stm) > 0:
                    mask = np.ma.getmaskarray(stm[0].data)
                    filled_gap_samples += int(mask.sum())
                    gap_reference_samples += int(mask.size)

                if len(sampling_rates) != 1:
                    raise ValueError("Cannot concatenate different sampling rates")
                data, dropped = chunk_samples(wf, previous_written_end)
                boundary_samples_dropped += dropped
                if len(data):
                    written_start = wf.stats.starttime + dropped * wf.stats.delta
                    chunk_map.append({
                        "output_start": int(total_points), "count": int(len(data)),
                        "start_utc": str(written_start), "end_utc": str(wf.stats.endtime),
                        "gap_seconds_before": (float(written_start - previous_written_end) - wf.stats.delta
                                               if previous_written_end is not None else 0.0),
                        "overlap_samples_dropped": dropped,
                    })
                    previous_written_end = wf.stats.endtime

                if ENABLE_GENERATION_REPORT:
                    update_data_stats(data_stats, data)
                    append_plot_preview(plot_preview, data, total_points)
                total_points += append_binary_chunk(raw_bin_file, data)
                chunks_written += 1
                chunk_start = chunk_end
                progress.update(1)

    if total_points == 0:
        tmp_path.unlink(missing_ok=True)
        print(f"Skipping {network}.{station}.{location}.{channel}: no samples written")
        return None

    final_path = Path(
        make_waveform_filename(
            output_dir, network, station, location, channel, total_points
        )
    )
    os.replace(tmp_path, final_path)
    print(f">> BinFile: Wrote {total_points} points to {final_path}")

    if ENABLE_GENERATION_REPORT:
        duration_seconds = (
            float(actual_end - actual_start)
            if actual_start is not None and actual_end is not None
            else None
        )
        fill_ratio = (
            filled_gap_samples / gap_reference_samples if gap_reference_samples else 0.0
        )
        sampling_rate_values = sorted(sampling_rates)
        original_sampling_rate_values = sorted(original_sampling_rates)
        downsample_factor_values = sorted(downsample_factors)
        delta_values = sorted(deltas)
        metadata = {
            "dataset": "maule",
            "kind": "waveform",
            "pipeline_version": PIPELINE_VERSION,
            "created_at_utc": current_utc_timestamp(),
            "binary_file": str(final_path),
            "binary_file_name": final_path.name,
            "binary_file_size_bytes": final_path.stat().st_size,
            "dtype": "float32",
            "number_of_points": int(total_points),
            "network": network,
            "station": station,
            "location": location,
            "channel": channel,
            "source": {
                "sds_root": SDS_ROOT,
                "requested_start": utc_datetime_to_string(t_start),
                "requested_end": utc_datetime_to_string(t_end),
                "actual_start": utc_datetime_to_string(actual_start),
                "actual_end": utc_datetime_to_string(actual_end),
                "actual_duration_seconds": duration_seconds,
            },
            "frequency": {
                "original_sampling_rates_hz": original_sampling_rate_values,
                "sampling_rates_hz": sampling_rate_values,
                "sampling_rate_hz": (
                    sampling_rate_values[0]
                    if len(sampling_rate_values) == 1
                    else None
                ),
                "delta_seconds_values": delta_values,
                "bandpass_min_hz": FMIN,
                "bandpass_max_hz": FMAX,
                "downsample_factor_requested": DOWNSAMPLE_FACTOR,
                "downsample_factors_applied": downsample_factor_values,
                "downsample_factor": (
                    downsample_factor_values[0]
                    if len(downsample_factor_values) == 1
                    else None
                ),
            },
            "signal_processing": {
                "gap_fill": {
                    "method": "Gaussian noise in masked gaps",
                    "noise_scale": 3.0,
                    "seed": 0,
                    "filled_samples": filled_gap_samples,
                    "reference_samples": gap_reference_samples,
                    "fill_ratio": fill_ratio,
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
                    "applied_factors": downsample_factor_values,
                    "method": (
                        "ObsPy Trace.decimate(no_filter=True) after bandpass "
                        "when freqmax is below the target Nyquist; otherwise "
                        "the trace remains at its original sampling rate"
                    ),
                },
                "trim": "actual available bounds for each loaded chunk",
            },
            "generation": {
                "chunk_days": GENERATION_CHUNK_DAYS,
                "max_days": GENERATION_MAX_DAYS,
                "chunks_total": total_chunks,
                "chunks_written": chunks_written,
                "chunks_empty": empty_chunks,
                "boundary_samples_dropped": boundary_samples_dropped,
                "chunk_map": chunk_map,
                "time_layout": "available chunks concatenated; unavailable intervals omitted and recorded in chunk_map",
                "entry_offset": GENERATION_ENTRY_OFFSET,
                "entry_stride": GENERATION_ENTRY_STRIDE,
            },
            "selection_filters": {
                "channel_prefixes": list(CHANNEL_PREFIXES),
                "networks": selected_values(SELECT_NETWORKS),
                "stations": selected_values(SELECT_STATIONS),
                "locations": selected_values(SELECT_LOCATIONS),
                "channels": selected_values(SELECT_CHANNELS),
            },
            "amplitude_stats": finalize_data_stats(data_stats),
        }
        write_waveform_generation_report(final_path, metadata, plot_preview)

    return final_path


def discover_sds_inventory(sds_root, channel_prefixes=CHANNEL_PREFIXES):
    """
    Walk an SDS archive directory tree and return all available
    (network, station, location, channel, start_time, end_time) tuples,
    where start_time / end_time reflect the first and last day-files
    actually present on disk.

    Only channels whose names start with one of `channel_prefixes` are
    included (default: broadband HH* and strong-motion HN*).
    """
    from pathlib import Path
    from collections import defaultdict

    sds_path = Path(sds_root)
    # Map (network, station, location, channel) -> list of (year, jday) pairs
    records = defaultdict(list)

    year_dirs = sorted(
        path for path in sds_path.iterdir() if path.is_dir() and path.name.isdigit()
    )
    for year_dir in tqdm(year_dirs, desc="SDS inventory years", unit="year"):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for net_dir in sorted(year_dir.iterdir()):
            if not net_dir.is_dir():
                continue
            network = net_dir.name
            if not matches_selected_network(network):
                continue
            for sta_dir in sorted(net_dir.iterdir()):
                if not sta_dir.is_dir():
                    continue
                station = sta_dir.name
                if not matches_selected_station(station):
                    continue
                for cha_dir in sorted(sta_dir.iterdir()):
                    if not cha_dir.is_dir() or not cha_dir.name.endswith(".D"):
                        continue
                    channel = cha_dir.name[:-2]  # strip ".D"
                    if not any(channel.startswith(p) for p in channel_prefixes):
                        continue
                    if not matches_selected_channel(channel):
                        continue
                    date_pairs_by_location = defaultdict(list)
                    for path in cha_dir.iterdir():
                        if not path.is_file():
                            continue
                        # SDS filename: NET.STA.LOC.CHA.D.YEAR.JDAY
                        parts = path.name.split(".")
                        if len(parts) < 7:
                            continue
                        location = parts[2]
                        if not matches_selected_location(location):
                            continue
                        date_pairs_by_location[location].append(
                            (int(parts[-2]), int(parts[-1]))
                        )
                    for location, date_pairs in date_pairs_by_location.items():
                        if not date_pairs:
                            continue
                        key = (network, station, location, channel)
                        records[key].extend(date_pairs)

    inventory = []
    for (network, station, location, channel), date_pairs in records.items():
        sorted_dates = sorted(date_pairs)
        first_year, first_jday = sorted_dates[0]
        last_year, last_jday = sorted_dates[-1]
        # start at the beginning of the first available day
        t_start = UTCDateTime(year=first_year, julday=first_jday)
        # end at the end of the last available day (exclusive upper bound)
        t_end = UTCDateTime(year=last_year, julday=last_jday) + SECONDS_PER_DAY
        inventory.append((network, station, location, channel, t_start, t_end))

    return sorted(inventory, key=lambda item: (item[0], item[1], item[2], item[3]))


def report_sds_inventory(sds_root, inventory):
    """
    Generate a comprehensive report of the SDS inventory.
    Loads metadata (sampling rate, data points) for each channel.
    """
    from collections import defaultdict

    print("\n" + "="*80)
    print("SDS ARCHIVE INVENTORY REPORT")
    print("="*80)

    # Collect metadata for each entry
    metadata = []
    for network, station, location, channel, t_start, t_end in inventory:
        try:
            st, stm, stf = load_and_fill_sds_traces(
                sds_root, t_start, t_end, network, station, location, channel
            )
            if len(stf) > 0 and len(st) > 0:
                tr = stf[0]
                trm = stm[0]
                sampling_rate = tr.stats.sampling_rate
                npts = tr.stats.npts
                actual_start = min(t.stats.starttime for t in st)
                actual_end = max(t.stats.endtime for t in st)
                duration_sec = actual_end - actual_start
                mask = np.ma.getmaskarray(trm.data)
                n_filled = int(mask.sum())
                fill_ratio = (n_filled / npts) if npts else 0.0
                metadata.append({
                    'network': network,
                    'station': station,
                    'location': location,
                    'channel': channel,
                    't_start': actual_start,
                    't_end': actual_end,
                    'duration_sec': duration_sec,
                    'sampling_rate': sampling_rate,
                    'npts': npts,
                    'n_filled': n_filled,
                    'fill_ratio': fill_ratio,
                })
        except Exception as e:
            print(f"Warning: Failed to load {network}.{station}.{location}.{channel}: {e}")

    if not metadata:
        print("No valid metadata collected.")
        return

    # Aggregation: by network, by station, by channel
    by_network = defaultdict(list)
    by_station = defaultdict(list)
    by_channel = defaultdict(list)

    for m in metadata:
        by_network[m['network']].append(m)
        by_station[(m['network'], m['station'])].append(m)
        by_channel[m['channel']].append(m)

    # Global statistics
    print(f"\nGLOBAL STATISTICS:")
    print(f"  Total channels collected: {len(metadata)}")
    print(f"  Unique networks: {len(by_network)} {list(by_network.keys())}")
    print(f"  Unique stations: {len(by_station)}")
    print(f"  Unique channels (types): {len(by_channel)} {sorted(by_channel.keys())}")

    all_t_starts = [m['t_start'] for m in metadata]
    all_t_ends = [m['t_end'] for m in metadata]
    global_start = min(all_t_starts)
    global_end = max(all_t_ends)
    global_duration = global_end - global_start

    print(f"  Global time range: {global_start.date} to {global_end.date}")
    print(f"  Total duration (max extent): {global_duration / SECONDS_PER_DAY:.1f} days ({global_duration / 3600:.1f} hours)")
    print(f"  Mean missing-sample fill ratio: {100.0 * np.mean([m['fill_ratio'] for m in metadata]):.2f}%")
    print(f"  Median missing-sample fill ratio: {100.0 * np.median([m['fill_ratio'] for m in metadata]):.2f}%")

    # Network-level statistics
    print(f"\nNETWORK STATISTICS:")
    for network in sorted(by_network.keys()):
        entries = by_network[network]
        unique_stations = len(set((m['station']) for m in entries))
        unique_channels = len(set((m['channel']) for m in entries))
        sample_rates = set(m['sampling_rate'] for m in entries)
        total_npts = sum(m['npts'] for m in entries)
        avg_npts = np.mean([m['npts'] for m in entries])
        avg_fill = np.mean([m['fill_ratio'] for m in entries])
        net_starts = [m['t_start'] for m in entries]
        net_ends = [m['t_end'] for m in entries]
        net_start = min(net_starts)
        net_end = max(net_ends)
        net_duration = net_end - net_start

        print(f"  {network}:")
        print(f"    Stations: {unique_stations}")
        print(f"    Channels: {unique_channels}")
        print(f"    Sampling rates: {sorted(sample_rates)}")
        print(f"    Data points per channel: min={min(m['npts'] for m in entries)}, "
              f"max={max(m['npts'] for m in entries)}, avg={avg_npts:.0f}")
        print(f"    Mean gap-fill ratio: {100.0 * avg_fill:.2f}%")
        print(f"    Total data points (all channels): {total_npts:.0f}")
        print(f"    Time range: {net_start.date} to {net_end.date}")
        print(f"    Duration: {net_duration / SECONDS_PER_DAY:.1f} days")

    # Station-level statistics (compact table)
    print(f"\nSTATION DETAILS (first 10):")
    print(f"{'Network':<8} {'Station':<10} {'Channels':<10} {'Sampling Rate (Hz)':<20} {'Avg Points':<15} {'Gap Fill %':<10} {'Start Date':<12} {'End Date':<12}")
    print("-" * 108)
    for i, (net_sta, entries) in enumerate(sorted(by_station.items())[:10]):
        network, station = net_sta
        n_channels = len(entries)
        sample_rates = sorted(set(m['sampling_rate'] for m in entries))
        avg_npts = np.mean([m['npts'] for m in entries])
        avg_fill = np.mean([m['fill_ratio'] for m in entries])
        st_start = min(m['t_start'] for m in entries).date
        st_end = max(m['t_end'] for m in entries).date
        sr_str = f"{sample_rates[0]:.1f}" if len(sample_rates) == 1 else f"{sample_rates}"
        print(f"{network:<8} {station:<10} {n_channels:<10} {sr_str:<20} {avg_npts:<15.0f} {100.0 * avg_fill:<10.2f} {str(st_start):<12} {str(st_end):<12}")
    if len(by_station) > 10:
        print(f"... and {len(by_station) - 10} more stations")

    # Channel (band) statistics
    print(f"\nCHANNEL TYPE STATISTICS:")
    for channel in sorted(by_channel.keys()):
        entries = by_channel[channel]
        n_entries = len(entries)
        sample_rates = sorted(set(m['sampling_rate'] for m in entries))
        total_npts = sum(m['npts'] for m in entries)
        avg_npts = np.mean([m['npts'] for m in entries])
        avg_fill = np.mean([m['fill_ratio'] for m in entries])
        print(f"  {channel}: {n_entries} entries, sampling rates: {sample_rates}, "
              f"avg points per entry: {avg_npts:.0f}, total points: {total_npts:.0f}, "
              f"mean gap-fill: {100.0 * avg_fill:.2f}%")

    print("\n" + "="*80 + "\n")








def main():
    print("Active selection filters:")
    print(f"  networks: {SELECT_NETWORKS}")
    print(f"  stations: {SELECT_STATIONS}")
    print(f"  locations: {SELECT_LOCATIONS}")
    print(f"  channels: {SELECT_CHANNELS}")

    # Discover all available stations, channels, and their actual time bounds
    # directly from the SDS directory tree (HH* and HN* channels only)
    inventory = discover_sds_inventory(SDS_ROOT)
    print(f"Discovered {len(inventory)} (network, station, channel) entries in SDS archive.")

    if selection_filters_enabled():
        inventory = filter_inventory(inventory)
        print(
            "After hardcoded filters: "
            f"{len(inventory)} (network, station, location, channel) entries."
        )

    if not inventory:
        print("No entries selected after applying filters. Nothing to process.")
        return

    if MAX_GENERATION_ENTRIES is not None and len(inventory) > MAX_GENERATION_ENTRIES:
        inventory = inventory[:MAX_GENERATION_ENTRIES]
        print(f"After max-entry cap: {len(inventory)} entries.")

    if GENERATION_ENTRY_STRIDE < 1:
        raise ValueError("MAULE_GENERATION_ENTRY_STRIDE must be >= 1")
    if GENERATION_ENTRY_OFFSET < 0:
        raise ValueError("MAULE_GENERATION_ENTRY_OFFSET must be >= 0")
    if GENERATION_ENTRY_STRIDE > 1 or GENERATION_ENTRY_OFFSET > 0:
        original_count = len(inventory)
        inventory = inventory[GENERATION_ENTRY_OFFSET::GENERATION_ENTRY_STRIDE]
        print(
            "After entry stride split: "
            f"{len(inventory)} of {original_count} entries "
            f"(offset={GENERATION_ENTRY_OFFSET}, stride={GENERATION_ENTRY_STRIDE})."
        )

    if not inventory:
        print("No entries assigned to this task. Nothing to process.")
        return

    if ENABLE_INVENTORY_REPORT:
        report_sds_inventory(SDS_ROOT, inventory)
    else:
        print("Inventory report disabled for fast run (ENABLE_INVENTORY_REPORT=False).")

    for network, station, location, channel, t_start, t_end in tqdm(
        inventory, desc="Waveform entries", unit="entry"
    ):
        tqdm.write(
            f"Processing {network}.{station}.{location}.{channel} "
            f"[{t_start.date} -> {t_end.date}]"
        )

        generate_waveform_binary(network, station, location, channel, t_start, t_end)


if __name__ == "__main__":
    main()
