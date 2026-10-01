from obspy import UTCDateTime
import json
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality import PIPELINE_VERSION, chunk_samples

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
    SECONDS_PER_DAY,
    clip_time_range,
    current_utc_timestamp,
    ensure_parent_directory,
    env_flag,
    filter_inventory,
    finalize_data_stats,
    load_and_fill_sds_traces,
    load_and_prepare_trace,
    make_data_stats,
    selected_values,
    selection_filters_enabled,
    tqdm,
    update_data_stats,
    update_time_bounds,
    utc_datetime_to_string,
)

DIRECTORY = os.getenv(
    "SEIFR_WAVEFORM_OUTPUT_DIR",
    "/path/to/data/seifr/waveforms/",
)
GENERATION_CHUNK_DAYS = int(os.getenv("SEIFR_GENERATION_CHUNK_DAYS", "1"))
_generation_max_days = os.getenv("SEIFR_GENERATION_MAX_DAYS")
GENERATION_MAX_DAYS = (
    int(_generation_max_days) if _generation_max_days is not None else None
)

ENABLE_GENERATION_REPORT = env_flag("SEIFR_ENABLE_WAVEFORM_REPORT", True)
SUMMARY_OUTPUT_DIR = Path(
    os.getenv(
        "SEIFR_SUMMARY_OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "metadata"),
    )
)
WAVEFORM_PLOT_MAX_POINTS = int(os.getenv("SEIFR_WAVEFORM_PLOT_MAX_POINTS", "200000"))
WAVEFORM_PLOT_POINTS_PER_CHUNK = int(
    os.getenv("SEIFR_WAVEFORM_PLOT_POINTS_PER_CHUNK", "4000")
)

REQUEST_START = UTCDateTime(os.getenv("SEIFR_START_DATE", "2020-1-1"))
REQUEST_DAYS = int(os.getenv("SEIFR_DAYS_TO_LOAD", str(365 * 2)))
REQUEST_END = REQUEST_START + REQUEST_DAYS * SECONDS_PER_DAY


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
    return str(filename), int(y.size)


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


def discover_sds_inventory(sds_root):
    from collections import defaultdict

    sds_path = Path(sds_root)
    records = defaultdict(list)

    year_dirs = sorted(
        path for path in sds_path.iterdir() if path.is_dir() and path.name.isdigit()
    )
    for year_dir in tqdm(year_dirs, desc="SDS inventory years", unit="year"):
        for net_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            network = net_dir.name
            if SELECT_NETWORKS is not None and network not in SELECT_NETWORKS:
                continue

            for sta_dir in sorted(path for path in net_dir.iterdir() if path.is_dir()):
                station = sta_dir.name
                if SELECT_STATIONS is not None and station not in SELECT_STATIONS:
                    continue

                for cha_dir in sorted(path for path in sta_dir.iterdir() if path.is_dir()):
                    if not cha_dir.name.endswith(".D"):
                        continue

                    channel = cha_dir.name[:-2]
                    if CHANNEL_PREFIXES and not any(
                        channel.startswith(prefix) for prefix in CHANNEL_PREFIXES
                    ):
                        continue
                    if SELECT_CHANNELS is not None and channel not in SELECT_CHANNELS:
                        continue

                    date_pairs_by_location = defaultdict(list)
                    for path in cha_dir.iterdir():
                        if not path.is_file():
                            continue

                        parts = path.name.split(".")
                        if len(parts) < 7:
                            continue

                        location = parts[2]
                        if SELECT_LOCATIONS is not None and location not in SELECT_LOCATIONS:
                            continue

                        try:
                            date_pairs_by_location[location].append(
                                (int(parts[-2]), int(parts[-1]))
                            )
                        except ValueError:
                            continue

                    for location, date_pairs in date_pairs_by_location.items():
                        if not date_pairs:
                            continue
                        records[(network, station, location, channel)].extend(date_pairs)

    inventory = []
    for (network, station, location, channel), date_pairs in records.items():
        sorted_dates = sorted(date_pairs)
        first_year, first_jday = sorted_dates[0]
        last_year, last_jday = sorted_dates[-1]
        actual_start = UTCDateTime(year=first_year, julday=first_jday)
        actual_end = UTCDateTime(year=last_year, julday=last_jday) + SECONDS_PER_DAY
        clipped_start, clipped_end = clip_time_range(
            actual_start, actual_end, REQUEST_START, REQUEST_END
        )
        if clipped_start is None:
            continue
        inventory.append(
            (network, station, location, channel, clipped_start, clipped_end)
        )

    return sorted(inventory, key=lambda item: (item[0], item[1], item[2], item[3]))


def report_sds_inventory(sds_root, inventory):
    from collections import defaultdict

    print("\n" + "=" * 80)
    print("SDS ARCHIVE INVENTORY REPORT")
    print("=" * 80)

    metadata = []
    for network, station, location, channel, t_start, t_end in inventory:
        try:
            st, stm, stf = load_and_fill_sds_traces(
                sds_root, t_start, t_end, network, station, location, channel
            )
            if len(stf) == 0 or len(st) == 0:
                continue

            tr = stf[0]
            trm = stm[0]
            actual_start = min(trace.stats.starttime for trace in st)
            actual_end = max(trace.stats.endtime for trace in st)
            mask = np.ma.getmaskarray(trm.data)
            missing = int(mask.sum())
            metadata.append(
                {
                    "network": network,
                    "station": station,
                    "location": location,
                    "channel": channel,
                    "t_start": actual_start,
                    "t_end": actual_end,
                    "duration_sec": actual_end - actual_start,
                    "sampling_rate": tr.stats.sampling_rate,
                    "npts": tr.stats.npts,
                    "fill_ratio": (missing / tr.stats.npts) if tr.stats.npts else 0.0,
                }
            )
        except Exception as exc:
            print(
                f"Warning: Failed to load {network}.{station}.{location}.{channel}: {exc}"
            )

    if not metadata:
        print("No valid metadata collected.")
        return

    by_network = defaultdict(list)
    by_station = defaultdict(list)
    by_channel = defaultdict(list)
    for item in metadata:
        by_network[item["network"]].append(item)
        by_station[(item["network"], item["station"])].append(item)
        by_channel[item["channel"]].append(item)

    print("\nGLOBAL STATISTICS:")
    print(f"  Total channels collected: {len(metadata)}")
    print(f"  Unique networks: {len(by_network)} {list(by_network.keys())}")
    print(f"  Unique stations: {len(by_station)}")
    print(f"  Unique channels (types): {len(by_channel)} {sorted(by_channel.keys())}")


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
    sampling_rates = set()
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
                st, stm, stf, chunk_actual_start, chunk_actual_end = load_and_prepare_trace(
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
                sampling_rates.add(float(wf.stats.sampling_rate))
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

                update_data_stats(data_stats, data)
                if ENABLE_GENERATION_REPORT:
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

    duration_seconds = (
        float(actual_end - actual_start)
        if actual_start is not None and actual_end is not None
        else None
    )
    fill_ratio = (
        filled_gap_samples / gap_reference_samples if gap_reference_samples else 0.0
    )
    sampling_rate_values = sorted(sampling_rates)
    delta_values = sorted(deltas)

    if ENABLE_GENERATION_REPORT:
        metadata = {
            "dataset": "seiFR",
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
                "sampling_rates_hz": sampling_rate_values,
                "sampling_rate_hz": (
                    sampling_rate_values[0]
                    if len(sampling_rate_values) == 1
                    else None
                ),
                "delta_seconds_values": delta_values,
                "bandpass_min_hz": FMIN,
                "bandpass_max_hz": FMAX,
                "downsample_factor": DOWNSAMPLE_FACTOR,
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
                    "factor": DOWNSAMPLE_FACTOR,
                    "method": "ObsPy decimate(no_filter=True) after anti-alias bandpass",
                },
                "trim": "actual available bounds for each loaded chunk",
            },
            "generation": {
                "chunk_days": GENERATION_CHUNK_DAYS,
                "max_days": GENERATION_MAX_DAYS,
                "requested_window_start": utc_datetime_to_string(REQUEST_START),
                "requested_window_days": REQUEST_DAYS,
                "requested_window_end": utc_datetime_to_string(REQUEST_END),
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

    print(
        f">> Summary: dataset=seiFR kind=waveform created_at={current_utc_timestamp()} "
        f"network={network} station={station} location={location} channel={channel} "
        f"points={total_points} requested_start={utc_datetime_to_string(t_start)} "
        f"requested_end={utc_datetime_to_string(t_end)} actual_start={utc_datetime_to_string(actual_start)} "
        f"actual_end={utc_datetime_to_string(actual_end)} sampling_rates={sampling_rate_values} "
        f"bandpass=({FMIN},{FMAX}) downsample_factor={DOWNSAMPLE_FACTOR} "
        f"fill_ratio={fill_ratio:.6f} gap_fill=gaussian_noise "
        f"selection_networks={selected_values(SELECT_NETWORKS)} selection_stations={selected_values(SELECT_STATIONS)} "
        f"selection_locations={selected_values(SELECT_LOCATIONS)} selection_channels={selected_values(SELECT_CHANNELS)} "
        f"boundary_samples_dropped={boundary_samples_dropped} chunks_empty={empty_chunks} "
        f"stats={finalize_data_stats(data_stats)}"
    )

    return final_path


def main():
    print("Active selection filters:")
    print(f"  networks: {SELECT_NETWORKS}")
    print(f"  stations: {SELECT_STATIONS}")
    print(f"  locations: {SELECT_LOCATIONS}")
    print(f"  channels: {SELECT_CHANNELS}")
    print(f"  channel prefixes: {CHANNEL_PREFIXES}")
    print(
        f"Requested time window: {REQUEST_START} -> {REQUEST_END} ({REQUEST_DAYS} days)"
    )
    print(f"Waveform report enabled: {ENABLE_GENERATION_REPORT}")

    inventory = discover_sds_inventory(SDS_ROOT)
    print(
        f"Discovered {len(inventory)} (network, station, location, channel) entries in SDS archive."
    )

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
        raise ValueError("SEIFR_GENERATION_ENTRY_STRIDE must be >= 1")
    if GENERATION_ENTRY_OFFSET < 0:
        raise ValueError("SEIFR_GENERATION_ENTRY_OFFSET must be >= 0")
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
