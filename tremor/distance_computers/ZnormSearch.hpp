#ifndef ZNORM_SEARCH_HPP
#define ZNORM_SEARCH_HPP

// Building blocks shared by TREMOR and the skip-sequential scan baseline:
// z-normalized squared Euclidean distance with SIMD early abandoning, and the
// canonical kNN selection with an exclusion zone (merge offset).

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <immintrin.h>
#include <iterator>
#include <set>
#include <utility>
#include <vector>

#include "../isax/iSAXTypes.hpp"

using idx_t = unsigned long long; // as in htremor/replication.hpp

constexpr float TREMOR_MISSING_VALUE = -100000.0f;

inline float cc_to_ed(float cc, int subsequence_length)
{
    return 2.0f * static_cast<float>(subsequence_length) * (1.0f - cc);
}

inline float ed_to_cc(float ed, int subsequence_length)
{
    return 1.0f - (ed / (2.0f * static_cast<float>(subsequence_length)));
}

inline void znormalize_inplace(ts_type *values, int len)
{
    if (values == nullptr || len <= 0)
        return;

    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < len; i++)
    {
        const double v = static_cast<double>(values[i]);
        sum += v;
        sum_sq += v * v;
    }

    const double mean = sum / static_cast<double>(len);
    double variance = (sum_sq / static_cast<double>(len)) - (mean * mean);
    if (variance < 0.0)
        variance = 0.0;
    const double stddev = std::sqrt(variance);

    if (stddev <= 1e-12)
    {
        for (int i = 0; i < len; i++)
            values[i] = 0.0f;
        return;
    }

    for (int i = 0; i < len; i++)
        values[i] = static_cast<ts_type>((static_cast<double>(values[i]) - mean) / stddev);
}

// Early abandoning is checked once per block of this many samples.
constexpr int ZNORM_BLOCK = 32;

// Block start offsets ordered by decreasing query energy (UCR-suite
// reordering): blocks where the z-normalized query is large contribute
// most to the distance of a non-matching window, so visiting them first
// lets the early-abandon test fire sooner. The tail that does not fill a
// whole block is handled separately by l2_dist_znorm_subsequence.
inline std::vector<int> query_block_order(const ts_type *query_znorm, int subsequence_length)
{
    const int blocks = subsequence_length / ZNORM_BLOCK;
    std::vector<std::pair<float, int>> energy(static_cast<size_t>(blocks));
    for (int b = 0; b < blocks; b++)
    {
        float e = 0.0f;
        for (int i = b * ZNORM_BLOCK; i < (b + 1) * ZNORM_BLOCK; i++)
            e += query_znorm[i] * query_znorm[i];
        energy[static_cast<size_t>(b)] = {e, b * ZNORM_BLOCK};
    }
    std::stable_sort(energy.begin(), energy.end(),
                     [](const std::pair<float, int> &a, const std::pair<float, int> &b)
                     { return a.first > b.first; });
    std::vector<int> order(static_cast<size_t>(blocks));
    for (int b = 0; b < blocks; b++)
        order[static_cast<size_t>(b)] = energy[static_cast<size_t>(b)].second;
    return order;
}

inline float hsum256(__m256 v)
{
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 h = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)));
}

// Squared Euclidean distance between the z-normalized query and the
// z-normalized window starting at local_start, abandoned (returning a value
// above `bound`) as soon as the partial sum exceeds `bound`. Whole blocks
// are visited in `block_order`; the remaining tail samples come last.
inline float l2_dist_znorm_subsequence(
    const ts_type *query_znorm,
    const float *raw_long_sequence,
    idx_t local_start,
    int subsequence_length,
    float mean,
    float stddev,
    float bound,
    const int *block_order,
    int *blocks_visited)
{
    if (query_znorm == nullptr || raw_long_sequence == nullptr || subsequence_length <= 0)
        return FLT_MAX;

    if (stddev <= 1e-12f)
    {
        // Pearson correlation is undefined for a constant window.
        // Match MASS: exclude it rather than reporting correlation 0.5.
        return FLT_MAX;
    }

    const float *raw = raw_long_sequence + local_start;
    const float inv_std = 1.0f / stddev;
    const __m256 mean_v = _mm256_set1_ps(mean);
    const __m256 inv_std_v = _mm256_set1_ps(inv_std);
    const __m256 missing_v = _mm256_set1_ps(TREMOR_MISSING_VALUE);
    const __m256 zero_v = _mm256_setzero_ps();
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    const int blocks = subsequence_length / ZNORM_BLOCK;
    for (int b = 0; b < blocks; b++)
    {
        const int off = block_order[b];
        for (int i = off; i < off + ZNORM_BLOCK; i += 16)
        {
            __m256 r0 = _mm256_loadu_ps(raw + i);
            __m256 r1 = _mm256_loadu_ps(raw + i + 8);
            r0 = _mm256_blendv_ps(r0, zero_v, _mm256_cmp_ps(r0, missing_v, _CMP_EQ_OQ));
            r1 = _mm256_blendv_ps(r1, zero_v, _mm256_cmp_ps(r1, missing_v, _CMP_EQ_OQ));
            const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(query_znorm + i), _mm256_mul_ps(_mm256_sub_ps(r0, mean_v), inv_std_v));
            const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(query_znorm + i + 8), _mm256_mul_ps(_mm256_sub_ps(r1, mean_v), inv_std_v));
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        }
        const float dist = hsum256(_mm256_add_ps(acc0, acc1));
        if (dist > bound)
        {
            *blocks_visited = b + 1;
            return dist;
        }
    }

    *blocks_visited = blocks;
    float dist = hsum256(_mm256_add_ps(acc0, acc1));
    for (int i = blocks * ZNORM_BLOCK; i < subsequence_length; i++)
    {
        const float raw_v = (raw[i] == TREMOR_MISSING_VALUE) ? 0.0f : raw[i];
        const float diff = query_znorm[i] - (raw_v - mean) * inv_std;
        dist += diff * diff;
    }
    return dist;
}

using KnnCandidate = std::pair<file_position_type, float>;

// Canonical exclusion-zone selection (DMASS implements the same).
inline std::vector<KnnCandidate> select_knn_with_exclusion(std::vector<KnnCandidate> candidates, int k, int merge_offset)
{
    std::sort(candidates.begin(), candidates.end(), [](const KnnCandidate &a, const KnnCandidate &b)
              { return a.second != b.second ? a.second < b.second : a.first < b.first; });
    std::vector<KnnCandidate> kept;
    std::set<file_position_type> kept_positions;
    for (const KnnCandidate &c : candidates)
    {
        if (static_cast<int>(kept.size()) == k)
            break;
        const auto next = kept_positions.lower_bound(c.first);
        if (next != kept_positions.end() && *next - c.first < static_cast<file_position_type>(merge_offset))
            continue;
        if (next != kept_positions.begin() && c.first - *std::prev(next) < static_cast<file_position_type>(merge_offset))
            continue;
        kept.push_back(c);
        kept_positions.insert(c.first);
    }
    return kept;
}

#endif
