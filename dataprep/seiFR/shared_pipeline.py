import os
import sys
import time
from pathlib import Path

import numpy as np
from obspy.clients.filesystem.sds import Client

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get("total")
            self.desc = kwargs.get("desc") or "progress"
            self.unit = kwargs.get("unit") or "it"
            self.count = 0
            self.postfix = ""
            self.last_print = 0.0
            if self.total is None and iterable is not None:
                try:
                    self.total = len(iterable)
                except TypeError:
                    self.total = None

        def __iter__(self):
            for item in self.iterable or []:
                yield item
                self.update(1)

        def __enter__(self):
            self._print(force=True)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._print(force=True, done=True)
            return False

        def update(self, n=1):
            self.count += n
            self._print()

        def set_postfix_str(self, s=None, refresh=True):
            self.postfix = s or ""
            if refresh:
                self._print()

        def _print(self, force=False, done=False):
            now = time.monotonic()
            if not force and now - self.last_print < 5.0:
                return
            self.last_print = now
            if self.total:
                progress = f"{self.count}/{self.total} {self.unit}"
            else:
                progress = f"{self.count} {self.unit}"
            suffix = f" | {self.postfix}" if self.postfix else ""
            state = " done" if done else ""
            print(f"{self.desc}: {progress}{suffix}{state}", file=sys.stderr)

        @staticmethod
        def write(message):
            print(message)


SECONDS_PER_DAY = 86400
SDS_ROOT = os.getenv("SEIFR_SDS_ROOT", "/path/to/raw/seifr/SDS/")
FMIN = float(os.getenv("SEIFR_FILTER_FMIN", "1"))
FMAX = float(os.getenv("SEIFR_FILTER_FMAX", "10"))
DOWNSAMPLE_FACTOR = int(os.getenv("SEIFR_DOWNSAMPLE_FACTOR", "2"))
_channel_prefixes = os.getenv("SEIFR_CHANNEL_PREFIXES", "BH")
CHANNEL_PREFIXES = tuple(
    prefix.strip() for prefix in _channel_prefixes.split(",") if prefix.strip()
)


_FALSE_ENV_VALUES = {"0", "false", "no", "off", ""}


def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


def parse_env_set(name):
    raw = os.getenv(name)
    if raw is None:
        return None
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return set(values) if values else None


SELECT_NETWORKS = parse_env_set("SEIFR_SELECT_NETWORKS")
SELECT_STATIONS = parse_env_set("SEIFR_SELECT_STATIONS")
SELECT_LOCATIONS = parse_env_set("SEIFR_SELECT_LOCATIONS")
SELECT_CHANNELS = parse_env_set("SEIFR_SELECT_CHANNELS")

ENABLE_INVENTORY_REPORT = env_flag("SEIFR_ENABLE_INVENTORY_REPORT", False)
_max_generation_entries = os.getenv("SEIFR_MAX_GENERATION_ENTRIES")
MAX_GENERATION_ENTRIES = (
    int(_max_generation_entries) if _max_generation_entries is not None else None
)
GENERATION_ENTRY_OFFSET = int(os.getenv("SEIFR_GENERATION_ENTRY_OFFSET", "0"))
GENERATION_ENTRY_STRIDE = int(os.getenv("SEIFR_GENERATION_ENTRY_STRIDE", "1"))
TEMPLATE_CHUNKSIZE = int(os.getenv("SEIFR_TEMPLATE_CHUNKSIZE", "100000"))
TEMPLATE_LOG_LIMIT = int(os.getenv("SEIFR_TEMPLATE_LOG_LIMIT", "20"))


def ensure_parent_directory(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def selected_values(values):
    return sorted(values) if values is not None else None


def current_utc_timestamp():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_datetime_to_string(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def matches_selection(network, station, location, channel):
    return (
        (SELECT_NETWORKS is None or network in SELECT_NETWORKS)
        and (SELECT_STATIONS is None or station in SELECT_STATIONS)
        and (SELECT_LOCATIONS is None or location in SELECT_LOCATIONS)
        and (SELECT_CHANNELS is None or channel in SELECT_CHANNELS)
    )


def filter_inventory(inventory):
    return [
        item
        for item in inventory
        if matches_selection(item[0], item[1], item[2], item[3])
    ]


def selection_filters_enabled():
    return any(
        value is not None
        for value in (
            SELECT_NETWORKS,
            SELECT_STATIONS,
            SELECT_LOCATIONS,
            SELECT_CHANNELS,
        )
    )


def clip_time_range(actual_start, actual_end, requested_start, requested_end):
    clipped_start = max(actual_start, requested_start)
    clipped_end = min(actual_end, requested_end)
    if clipped_start >= clipped_end:
        return None, None
    return clipped_start, clipped_end


def load_and_fill_sds_traces(
    sds_root,
    t0,
    t1,
    network,
    station,
    location,
    channel,
    noise_scale=3.0,
    seed=0,
):
    sds = Client(sds_root)
    st = sds.get_waveforms(
        network, station, location, channel, starttime=t0, endtime=t1
    )

    stm = st.copy()
    stm.merge(fill_value=None)

    if len(stm) == 0:
        return st, stm, stm.copy()

    tr = stm[0]
    x = tr.data
    mask = np.ma.getmaskarray(x)
    n_missing = int(mask.sum())

    stf = stm.copy()
    if n_missing:
        rng = np.random.default_rng(seed)
        observed = x.compressed()
        std = float(observed.std()) if observed.size else 1.0
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        x_filled = x.filled(0.0).astype(np.float32)
        x_filled[mask] = rng.normal(
            0.0, noise_scale * std, n_missing
        ).astype(np.float32)
        stf[0].data = x_filled

    return st, stm, stf


def resolve_actual_bounds(stream):
    return (
        min(trace.stats.starttime for trace in stream),
        max(trace.stats.endtime for trace in stream),
    )


def apply_standard_preprocessing(stream, starttime, endtime):
    if DOWNSAMPLE_FACTOR < 1:
        raise ValueError("SEIFR_DOWNSAMPLE_FACTOR must be >= 1")

    stream.detrend("constant")
    stream.detrend("linear")
    stream.taper(0.02, type="cosine")
    stream.filter("bandpass", freqmin=FMIN, freqmax=FMAX, zerophase=True)
    stream.trim(starttime=starttime, endtime=endtime)
    if DOWNSAMPLE_FACTOR > 1:
        for trace in stream:
            target_rate = float(trace.stats.sampling_rate) / float(DOWNSAMPLE_FACTOR)
            target_nyquist = target_rate / 2.0
            if FMAX >= target_nyquist:
                raise ValueError(
                    f"Bandpass max {FMAX} Hz must be below target Nyquist "
                    f"{target_nyquist} Hz for downsample factor {DOWNSAMPLE_FACTOR}"
                )
        stream.decimate(
            factor=DOWNSAMPLE_FACTOR,
            no_filter=True,
            strict_length=False,
        )
    return stream


def load_and_prepare_trace(
    sds_root,
    t0,
    t1,
    network,
    station,
    location,
    channel,
):
    st, stm, stf = load_and_fill_sds_traces(
        sds_root,
        t0,
        t1,
        network,
        station,
        location,
        channel,
    )

    if len(st) == 0 or len(stf) == 0:
        return st, stm, stf, None, None

    actual_start, actual_end = resolve_actual_bounds(st)
    stf = apply_standard_preprocessing(stf, actual_start, actual_end)
    return st, stm, stf, actual_start, actual_end


def update_time_bounds(current_start, current_end, chunk_start, chunk_end):
    if chunk_start is not None:
        current_start = (
            chunk_start if current_start is None else min(current_start, chunk_start)
        )
    if chunk_end is not None:
        current_end = chunk_end if current_end is None else max(current_end, chunk_end)
    return current_start, current_end


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
