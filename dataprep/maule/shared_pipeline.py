import os
from pathlib import Path
import sys
import time

from obspy.clients.filesystem.sds import Client

import numpy as np

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


SDS_ROOT = os.getenv("MAULE_SDS_ROOT", "/path/to/raw/maule/SDS/")
FMIN = float(os.getenv("MAULE_FILTER_FMIN", "1"))
FMAX = float(os.getenv("MAULE_FILTER_FMAX", "10"))
DOWNSAMPLE_FACTOR = int(os.getenv("MAULE_DOWNSAMPLE_FACTOR", "2"))
CHANNEL_PREFIXES = ("HH", "HN")

# Optional hardcoded filters shared by dataset and query generation.
# Defaults select a network, then include all stations/sections, locations,
# and HH/HN channels found for that network.
# Set any of these to None to expand scope.
# Example:
# SELECT_NETWORKS = {"XS", "ZE"}
# SELECT_STATIONS = {"QF02B", "QC01", "G01S"}
# SELECT_LOCATIONS = {"", "00"}
# SELECT_CHANNELS = {"HNE", "HNN", "HNZ", "HHE", "HHN", "HHZ"}
SELECT_NETWORKS = {"XS", "ZE"}
SELECT_STATIONS = None
SELECT_LOCATIONS = None
SELECT_CHANNELS = None

# Optional small-run controls for faster iteration.
# Set ENABLE_INVENTORY_REPORT=True for the full metadata summary.
# Set MAULE_MAX_GENERATION_ENTRIES to cap the selected entries for testing.
ENABLE_INVENTORY_REPORT = False
_max_generation_entries = os.getenv("MAULE_MAX_GENERATION_ENTRIES")
MAX_GENERATION_ENTRIES = (
    int(_max_generation_entries) if _max_generation_entries is not None else None
)
GENERATION_ENTRY_OFFSET = int(os.getenv("MAULE_GENERATION_ENTRY_OFFSET", "0"))
GENERATION_ENTRY_STRIDE = int(os.getenv("MAULE_GENERATION_ENTRY_STRIDE", "1"))
TEMPLATE_CHUNKSIZE = int(os.getenv("MAULE_TEMPLATE_CHUNKSIZE", "100000"))
TEMPLATE_LOG_LIMIT = int(os.getenv("MAULE_TEMPLATE_LOG_LIMIT", "20"))


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
    plot=False,
):
    """Load SDS traces and fill only masked gaps with Gaussian noise."""
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
        std = x.compressed().std()
        x_filled = x.filled(0.0).astype(np.float32)
        x_filled[mask] = rng.normal(
            0.0, noise_scale * std, n_missing
        ).astype(np.float32)
        stf[0].data = x_filled

    if plot:
        tr.plot()
        stf[0].plot()

    return st, stm, stf


def matches_selection(network, station, location, channel):
    return (
        (SELECT_NETWORKS is None or network in SELECT_NETWORKS)
        and (SELECT_STATIONS is None or station in SELECT_STATIONS)
        and (SELECT_LOCATIONS is None or location in SELECT_LOCATIONS)
        and (SELECT_CHANNELS is None or channel in SELECT_CHANNELS)
    )


def matches_selected_network(network):
    return SELECT_NETWORKS is None or network in SELECT_NETWORKS


def matches_selected_station(station):
    return SELECT_STATIONS is None or station in SELECT_STATIONS


def matches_selected_location(location):
    return SELECT_LOCATIONS is None or location in SELECT_LOCATIONS


def matches_selected_channel(channel):
    return SELECT_CHANNELS is None or channel in SELECT_CHANNELS


def filter_inventory(inventory):
    return [
        item for item in inventory if matches_selection(item[0], item[1], item[2], item[3])
    ]


def selection_filters_enabled():
    return any(
        x is not None
        for x in (
            SELECT_NETWORKS,
            SELECT_STATIONS,
            SELECT_LOCATIONS,
            SELECT_CHANNELS,
        )
    )


def resolve_actual_bounds(stream):
    return (
        min(trace.stats.starttime for trace in stream),
        max(trace.stats.endtime for trace in stream),
    )


def get_trace_downsample_factor(trace):
    return int(getattr(trace.stats, "tremor_downsample_factor", 1))


def get_trace_original_sampling_rate(trace):
    return float(
        getattr(
            trace.stats,
            "tremor_original_sampling_rate_hz",
            float(trace.stats.sampling_rate) * get_trace_downsample_factor(trace),
        )
    )


def apply_standard_preprocessing(stream, starttime, endtime):
    if DOWNSAMPLE_FACTOR < 1:
        raise ValueError("MAULE_DOWNSAMPLE_FACTOR must be >= 1")

    for trace in stream:
        original_rate = float(trace.stats.sampling_rate)
        original_nyquist = original_rate / 2.0
        if FMAX >= original_nyquist:
            raise ValueError(
                f"Bandpass max {FMAX} Hz must be below original Nyquist "
                f"{original_nyquist} Hz for sampling rate {original_rate} Hz"
            )
        trace.stats.tremor_original_sampling_rate_hz = original_rate
        trace.stats.tremor_requested_downsample_factor = DOWNSAMPLE_FACTOR
        trace.stats.tremor_downsample_factor = 1

    stream.detrend("constant")
    stream.detrend("linear")
    stream.taper(0.02, type="cosine")
    stream.filter("bandpass", freqmin=FMIN, freqmax=FMAX, zerophase=True)
    stream.trim(starttime=starttime, endtime=endtime)

    if DOWNSAMPLE_FACTOR > 1:
        for trace in stream:
            original_rate = get_trace_original_sampling_rate(trace)
            target_rate = original_rate / float(DOWNSAMPLE_FACTOR)
            target_nyquist = target_rate / 2.0
            if FMAX < target_nyquist:
                trace.decimate(
                    factor=DOWNSAMPLE_FACTOR,
                    no_filter=True,
                    strict_length=False,
                )
                trace.stats.tremor_downsample_factor = DOWNSAMPLE_FACTOR

    return stream


def ensure_parent_directory(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


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
