"""Shared invariants for waveform concatenation and fixed-duration queries."""
import math
import numpy as np

PIPELINE_VERSION = 2


def chunk_samples(trace, previous_end):
    """Discard only samples whose timestamps already occurred in the output."""
    drop = 0
    if previous_end is not None:
        overlap = float(previous_end - trace.stats.starttime) * trace.stats.sampling_rate
        if overlap >= -1e-6:
            drop = min(len(trace.data), int(math.floor(overlap + 1e-6)) + 1)
    return trace.data[drop:], drop


def complete_template(trace, merged_stream, requested_start, requested_end):
    """Return the intended fixed-length window, or None for unavailable data.

    The inclusive SDS endpoint is excluded; the duration in samples is rounded
    down to a multiple of eight for the existing SIMD/PAA search interfaces.
    One output sample of endpoint tolerance accommodates sample-grid rounding.
    """
    rate = float(trace.stats.sampling_rate)
    length = int(round(float(requested_end - requested_start) * rate)) // 8 * 8
    tolerance = 1.01 / rate
    if (length <= 0 or len(trace.data) < length
            or float(trace.stats.starttime - requested_start) > tolerance
            or float(requested_end - trace.stats.endtime) > tolerance
            or any(np.ma.getmaskarray(tr.data).any() for tr in merged_stream)):
        return None
    data = np.asarray(trace.data[:length], dtype=np.float32)
    if not np.isfinite(data).all() or float(np.std(data, dtype=np.float64)) == 0:
        return None
    return data.copy()
