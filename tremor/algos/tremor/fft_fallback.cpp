#include "fft_fallback.hpp"

#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace tremor_fft
{
    State &state()
    {
        static State s;
        return s;
    }

    static double now()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // MemAvailable of /proc/meminfo, in bytes (0 if unknown).
    static double available_memory()
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        double kb = 0.0;
        std::string unit;
        while (meminfo >> key >> kb >> unit)
            if (key == "MemAvailable:")
                return kb * 1024.0;
        return 0.0;
    }

    void prepare_query(const ts_type *query)
    {
        State &s = state();
        double sum = 0.0, squares = 0.0;
        for (int i = 0; i < s.block; i++)
        {
            const double v = i < s.m ? static_cast<double>(query[s.m - 1 - i]) : 0.0; // reversed query
            s.query_block[i] = v;
            sum += v;
            squares += v * v;
        }
        s.query_sum = sum;
        s.query_squares = squares;
        s.query_norm = std::sqrt(squares);
        fftw_execute_dft_r2c(s.forward, s.query_block, s.query_spectrum);
    }

    // Seconds of an FFT pass (no window passes the filter), and of the scan kernel per window when it abandons after
    // the first block and when it visits every block, with the first window of the chunk as the template.
    static void calibrate(int threads, int rank)
    {
        State &s = state();
        std::vector<ts_type> query(static_cast<size_t>(s.m));
        idx_t probe = 0;
        while (probe + 1 < s.windows && s.stds[probe] <= 1e-12f)
            probe++;
        for (int i = 0; i < s.m; i++)
        {
            const float v = s.raw[s.first_raw + probe + static_cast<idx_t>(i)];
            query[static_cast<size_t>(i)] = v == TREMOR_MISSING_VALUE ? 0.0f : v;
        }
        znormalize_inplace(query.data(), s.m);
        const std::vector<int> order = query_block_order(query.data(), s.m);
        s.blocks_per_window = std::max(1, s.m / ZNORM_BLOCK);

        prepare_query(query.data());
        volatile long long cursor = 0;
        double start = now();
#pragma omp parallel num_threads(threads)
        pass(&cursor, [] {}, []
             { return -FLT_MAX; }, [](idx_t, idx_t) {});
        s.fft_seconds = now() - start;

        const long long sample = std::min<long long>(static_cast<long long>(s.windows), 1LL << 24);
        double times[2] = {0.0, 0.0};
        static volatile double sink = 0.0; // keeps the calibration loops from being optimized away
        for (int full = 0; full < 2; full++)
        {
            const float bound = full ? FLT_MAX : 0.0f;
            double checksum = 0.0;
            start = now();
#pragma omp parallel for num_threads(threads) schedule(static, 1 << 16) reduction(+ : checksum)
            for (long long id = 0; id < sample; id++)
            {
                int blocks = 0;
                const idx_t raw_start = s.first_raw + static_cast<idx_t>(id);
                if (raw_start + static_cast<idx_t>(s.m) <= s.chunk_length && s.stds[id] > 1e-12f)
                    checksum += l2_dist_znorm_subsequence(query.data(), s.raw, raw_start, s.m, s.means[id], s.stds[id],
                                                          bound, order.data(), &blocks) +
                                blocks;
            }
            times[full] = (now() - start) / static_cast<double>(std::max(1LL, sample));
            sink = sink + checksum;
        }
        s.scan_first_block = times[0];
        s.scan_per_block = std::max(0.0, (times[1] - times[0]) / std::max(1, s.blocks_per_window - 1));
        std::printf("[Node %d] FFT fallback: %lld pieces of %d points; pass %.3f s; scan %.3f s (1 block) to %.3f s "
                    "(%d blocks) per pass\n",
                    rank, s.pieces, s.block, s.fft_seconds, scan_seconds(1.0), scan_seconds(s.blocks_per_window),
                    s.blocks_per_window);
    }

    void build(const float *raw, idx_t chunk_length, idx_t first_raw, idx_t windows, int m, const float *means,
               const float *stds, int threads, int rank)
    {
        State &s = state();
        if (windows == 0 || m <= 0)
            return;
        s.raw = raw;
        s.chunk_length = chunk_length;
        s.first_raw = first_raw;
        s.windows = windows;
        s.m = m;
        s.means = means;
        s.stds = stds;
        s.block = 1 << 15;
        while (s.block < 4 * m)
            s.block *= 2;
        s.step = s.block - m + 1;
        s.pieces = static_cast<long long>((windows + static_cast<idx_t>(s.step) - 1) / static_cast<idx_t>(s.step));
        const int bins = s.block / 2 + 1;
        s.stride = (bins + 3) & ~3; // keeps every piece's spectrum aligned
        const double bytes = static_cast<double>(s.pieces) * s.stride * sizeof(fftw_complex);
        const double free_bytes = available_memory();
        if (free_bytes > 0.0 && bytes > 0.5 * free_bytes)
        {
            std::printf("[Node %d] FFT fallback disabled: spectra need %.1f GB, %.1f GB available\n", rank, bytes / 1e9,
                        free_bytes / 1e9);
            return;
        }
        s.spectra = fftw_alloc_complex(static_cast<size_t>(s.pieces) * s.stride);
        s.query_spectrum = fftw_alloc_complex(static_cast<size_t>(s.stride));
        s.query_block = fftw_alloc_real(static_cast<size_t>(s.block));
        double *scratch = fftw_alloc_real(static_cast<size_t>(s.block));
        if (s.spectra == nullptr || s.query_spectrum == nullptr || s.query_block == nullptr || scratch == nullptr)
        {
            std::printf("[Node %d] FFT fallback disabled: allocation failed\n", rank);
            return;
        }
        // Single-threaded plans, executed on different pieces by different threads (FFTW_ESTIMATE: arrays untouched).
        s.forward = fftw_plan_dft_r2c_1d(s.block, scratch, s.query_spectrum, FFTW_ESTIMATE);
        s.backward = fftw_plan_dft_c2r_1d(s.block, s.query_spectrum, scratch, FFTW_ESTIMATE);
        fftw_free(scratch);
        s.piece_norm.assign(static_cast<size_t>(s.pieces), 0.0);

#pragma omp parallel num_threads(threads)
        {
            double *piece = fftw_alloc_real(static_cast<size_t>(s.block));
#pragma omp for schedule(static)
            for (long long c = 0; c < s.pieces; c++)
            {
                const idx_t start = first_raw + static_cast<idx_t>(c) * s.step;
                double squares = 0.0;
                for (int i = 0; i < s.block; i++)
                {
                    const idx_t p = start + static_cast<idx_t>(i);
                    const double v = (p < chunk_length && raw[p] != TREMOR_MISSING_VALUE) ? raw[p] : 0.0;
                    piece[i] = v;
                    squares += v * v;
                }
                s.piece_norm[static_cast<size_t>(c)] = std::sqrt(squares);
                fftw_execute_dft_r2c(s.forward, piece, s.spectra + static_cast<size_t>(c) * s.stride);
            }
            fftw_free(piece);
        }
        s.ready = true;
        calibrate(threads, rank);
    }
}
