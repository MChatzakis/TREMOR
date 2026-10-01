#ifndef TREMOR_FFT_FALLBACK_HPP
#define TREMOR_FFT_FALLBACK_HPP

// FFT fallback: when the bound of a template is too loose for early abandoning to pay off (e.g., kNN whose answers
// are far from the template), the sliding dot products of the template with every window of this rank are cheaper
// to compute with FFTs than window by window. As in MASS V3, the owned windows are split into pieces of `block`
// points that overlap by m - 1 points; the spectrum of every piece is computed once, at index construction, and a
// template costs one inverse FFT per piece. The FFT distances only filter the windows: every window whose FFT
// distance may be within the bound (allowing for the FFT error) is verified with the exact kernel, so the answers
// are exactly those of the scan.

#include <fftw3.h>

#include <atomic>
#include <cmath>
#include <vector>

#include "../../distance_computers/ZnormSearch.hpp"

namespace tremor_fft
{
    struct State
    {
        bool ready = false;
        const float *raw = nullptr; // local chunk
        idx_t chunk_length = 0;
        idx_t first_raw = 0; // local raw start of owned window 0
        idx_t windows = 0;   // owned windows
        const float *stds = nullptr;
        const float *means = nullptr;
        int m = 0, block = 0, step = 0, stride = 0;
        long long pieces = 0;
        fftw_complex *spectra = nullptr; // pieces x stride bins
        std::vector<double> piece_norm;  // Euclidean norm of every piece's samples (FFT error bound)
        fftw_plan forward = nullptr, backward = nullptr;
        fftw_complex *query_spectrum = nullptr;
        double *query_block = nullptr; // reversed, zero-padded query
        double query_sum = 0.0, query_squares = 0.0, query_norm = 0.0;
        double fft_seconds = 0.0;      // one pass over all windows (calibrated)
        double scan_first_block = 0.0; // scan seconds per window abandoned after one block (calibrated)
        double scan_per_block = 0.0;   // and per further block
        int blocks_per_window = 1;
    };

    State &state();

    // At the end of index construction: the spectra of the pieces and the cost calibration; no-op when the
    // spectra do not fit in memory.
    void build(const float *raw, idx_t chunk_length, idx_t first_raw, idx_t windows, int m, const float *means,
               const float *stds, int threads, int rank);

    inline bool available() { return state().ready; }

    // Estimated seconds of a scan that visits `blocks` blocks per window on average, and of an FFT pass.
    inline double scan_seconds(double blocks)
    {
        const State &s = state();
        return static_cast<double>(s.windows) * (s.scan_first_block + std::max(0.0, blocks - 1.0) * s.scan_per_block);
    }
    inline double fft_seconds() { return state().fft_seconds; }

    // Per template, by one thread: the spectrum of the reversed z-normalized query.
    void prepare_query(const ts_type *query);

    // Every calling thread claims pieces from *cursor; on_piece() runs once per claimed piece, and visit(id, raw_start)
    // runs for every owned window whose FFT distance may be within bound().
    template <class OnPiece, class Bound, class Visit>
    void pass(volatile long long *cursor, OnPiece on_piece, Bound bound, Visit visit)
    {
        const State &s = state();
        thread_local fftw_complex *product = nullptr;
        thread_local double *dots = nullptr;
        thread_local int allocated = 0;
        if (allocated != s.block)
        {
            fftw_free(product);
            fftw_free(dots);
            product = fftw_alloc_complex(static_cast<size_t>(s.stride));
            dots = fftw_alloc_real(static_cast<size_t>(s.block));
            allocated = s.block;
        }
        const int bins = s.block / 2 + 1;
        const double scale = 1.0 / s.block, m = s.m;
        const double margin = 2e-4 * m; // float kernel vs. double
        const double error = 1e2 * 2.220446049250313e-16 * std::log2(static_cast<double>(s.block)) * s.query_norm;
        for (;;)
        {
            const long long c = __atomic_fetch_add(cursor, 1LL, __ATOMIC_RELAXED);
            if (c >= s.pieces)
                break;
            on_piece();
            const fftw_complex *x = s.spectra + static_cast<size_t>(c) * s.stride;
            const fftw_complex *y = s.query_spectrum;
            for (int i = 0; i < bins; i++)
            {
                product[i][0] = x[i][0] * y[i][0] - x[i][1] * y[i][1];
                product[i][1] = x[i][1] * y[i][0] + x[i][0] * y[i][1];
            }
            fftw_execute_dft_c2r(s.backward, product, dots);
            const double dot_error = error * s.piece_norm[static_cast<size_t>(c)]; // bound on |FFT dot - exact dot|
            const idx_t first = static_cast<idx_t>(c) * s.step;
            const idx_t last = std::min<idx_t>(first + s.step, s.windows);
            for (idx_t id = first; id < last; id++)
            {
                const idx_t raw_start = s.first_raw + id;
                if (raw_start + static_cast<idx_t>(s.m) > s.chunk_length || s.raw[raw_start] == TREMOR_MISSING_VALUE)
                    continue;
                const double sd = s.stds[id];
                if (sd <= 1e-12)
                    continue; // the kernel excludes constant windows
                const double dot = dots[s.m - 1 + (id - first)] * scale;
                const double dist = s.query_squares + m - 2.0 * (dot - s.means[id] * s.query_sum) / sd;
                if (dist - 2.0 * dot_error / sd - margin <= bound())
                    visit(id, raw_start);
            }
        }
    }
}

#endif
