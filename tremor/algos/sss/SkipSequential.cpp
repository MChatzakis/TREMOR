#include "SkipSequential.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#include "../../isax/SAX.hpp"
#include "../../isax/iSAXPqueue.hpp"

namespace
{
    // iSAX settings of TREMOR's index: 8-bit symbols for every PAA segment.
    constexpr int SAX_BIT_CARDINALITY = 8;
    constexpr int SAX_ALPHABET = 1 << SAX_BIT_CARDINALITY;
    constexpr long long RANGE = 1LL << 16; // subsequences per work unit of the scan

    double resident_bytes()
    {
        long pages = 0, resident = 0;
        if (FILE *f = std::fopen("/proc/self/statm", "r"))
        {
            if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2)
                resident = 0;
            std::fclose(f);
        }
        return static_cast<double>(resident) * static_cast<double>(sysconf(_SC_PAGESIZE));
    }


    void read_samples(const std::string &filename, idx_t first, idx_t count, float *out)
    {
        FILE *f = std::fopen(filename.c_str(), "rb");
        if (f == nullptr || fseeko(f, static_cast<off_t>(first) * sizeof(float), SEEK_SET) != 0 ||
            std::fread(out, sizeof(float), count, f) != count)
            throw std::runtime_error("cannot read " + filename);
        std::fclose(f);
    }
}

SkipSequentialScan::SkipSequentialScan(int threads, int paa_segments, int merge_offset, int replication_groups,
                                       bool lower_bounds)
    : threads(threads), paa_segments(paa_segments), merge_offset(merge_offset), lower_bounds(lower_bounds),
      cardinalities(static_cast<size_t>(paa_segments), SAX_BIT_CARDINALITY)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    // Same groups as TREMOR: ranks / groups consecutive ranks each, the last group takes the rest.
    groups = replication_groups == 0 ? ranks : replication_groups;
    if (groups < 1 || groups > ranks)
        throw std::invalid_argument("--replication-groups must be between 0 and the number of ranks");
    group = std::min(rank / (ranks / groups), groups - 1);
    MPI_Comm_split(MPI_COMM_WORLD, group, rank, &group_comm);
    MPI_Comm_size(group_comm, &group_size);
    MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, group_comm, &counter, &counter_window);
    *counter = 0;
    MPI_Barrier(group_comm);
    MPI_Win_lock_all(0, counter_window);
    // The other ranks' claims on the counter complete only while its rank runs MPI's
    // progress engine; without this thread they wait until it finishes its template.
    int group_rank = 0;
    MPI_Comm_rank(group_comm, &group_rank);
    if (group_size > 1 && group_rank == 0)
        progress = std::thread([this] {
            while (!stop_progress)
            {
                int flag = 0;
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, group_comm, &flag, MPI_STATUS_IGNORE);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
}

SkipSequentialScan::~SkipSequentialScan()
{
    stop_progress = true;
    if (progress.joinable())
        progress.join();
    MPI_Barrier(group_comm);
    MPI_Win_unlock_all(counter_window);
    MPI_Win_free(&counter_window);
    MPI_Comm_free(&group_comm);
}

int SkipSequentialScan::next_query(int query_count)
{
    if (group_size == 1)
        return std::min((*counter)++, query_count);
    const int one = 1;
    int claimed = 0;
    MPI_Fetch_and_op(&one, &claimed, MPI_INT, 0, 0, MPI_SUM, counter_window);
    MPI_Win_flush(0, counter_window);
    return std::min(claimed, query_count);
}

void SkipSequentialScan::build(const std::string &filename, idx_t total_samples, int window_length)
{
    window = window_length;
    // Same split as TREMOR and D-MASS: group g owns the subsequences that start in the g-th
    // slice of total_samples / groups samples; the last group also owns the remainder.
    const idx_t total = total_samples - window + 1; // subsequences of the whole waveform
    const idx_t slice = total_samples / groups;
    first_subsequence = slice * group;
    subsequences = (group == groups - 1 ? total : std::min(first_subsequence + slice, total)) - first_subsequence;

    raw.resize(subsequences + window - 1);
    read_samples(filename, first_subsequence, raw.size(), raw.data());
    const double resident_after_load = resident_bytes();

    // Exact two-pass statistics and the iSAX summary of every subsequence (as in TREMOR's index);
    // the sequential scan keeps only the statistics.
    means.assign(subsequences, 0.0f);
    stds.assign(subsequences, 0.0f);
    sax.assign(lower_bounds ? subsequences * paa_segments : 0, 0);
#pragma omp parallel num_threads(threads)
    {
        std::vector<ts_type> normalized(static_cast<size_t>(window));
#pragma omp for schedule(static)
        for (long long id = 0; id < static_cast<long long>(subsequences); id++)
        {
            const float *w = &raw[id];
            if (w[0] == TREMOR_MISSING_VALUE)
                continue; // not searched, as in TREMOR
            double sum = 0.0;
            for (int d = 0; d < window; d++)
                sum += w[d] == TREMOR_MISSING_VALUE ? 0.0 : static_cast<double>(w[d]);
            const double mean = sum / window;
            double squares = 0.0;
            for (int d = 0; d < window; d++)
            {
                const double v = (w[d] == TREMOR_MISSING_VALUE ? 0.0 : static_cast<double>(w[d])) - mean;
                squares += v * v;
            }
            const double stddev = std::sqrt(squares / window);
            const bool degenerate = stddev <= 1e-12;
            means[id] = static_cast<float>(mean);
            stds[id] = static_cast<float>(degenerate ? 0.0 : stddev);
            if (!lower_bounds)
                continue;
            for (int d = 0; d < window; d++)
            {
                const double v = w[d] == TREMOR_MISSING_VALUE ? 0.0 : static_cast<double>(w[d]);
                normalized[d] = static_cast<ts_type>((v - mean) / (degenerate ? 1.0 : stddev));
            }
            sax_from_ts(normalized.data(), &sax[id * paa_segments], window / paa_segments, paa_segments,
                        SAX_ALPHABET, SAX_BIT_CARDINALITY);
        }
    }
    summary_bytes = resident_bytes() - resident_after_load;
}

float SkipSequentialScan::lower_bound(ts_type *query_paa, idx_t id) const
{
    // Same lower bound as TREMOR's record-level check (SIMD version for 16 segments).
    sax_type *word = const_cast<sax_type *>(&sax[id * paa_segments]);
    sax_type *card = const_cast<sax_type *>(cardinalities.data());
    const float ratio = static_cast<float>(window) / static_cast<float>(paa_segments);
    if (paa_segments == 16)
        return minidist_paa_to_isax_rawa_SIMD(query_paa, word, card, SAX_BIT_CARDINALITY, SAX_ALPHABET,
                                              paa_segments, MINVAL, MAXVAL, ratio);
    return minidist_paa_to_isax(query_paa, word, card, SAX_BIT_CARDINALITY, SAX_ALPHABET,
                                paa_segments, MINVAL, MAXVAL, ratio);
}

void SkipSequentialScan::prepare_query(const float *query, std::vector<ts_type> &znorm, std::vector<ts_type> &paa,
                                       std::vector<int> &order) const
{
    znorm.assign(query, query + window);
    znormalize_inplace(znorm.data(), window);
    paa.resize(static_cast<size_t>(paa_segments));
    paa_from_ts(znorm.data(), paa.data(), paa_segments, window / paa_segments);
    order = query_block_order(znorm.data(), window);
}

void SkipSequentialScan::search_threshold(const float *queries, int query_count, float correlation,
                                          const std::string &output)
{
    const float bound = cc_to_ed(correlation, window);
    FILE *out = std::fopen((output + "_" + std::to_string(rank) + ".csv").c_str(), "w");
    if (out == nullptr)
        throw std::runtime_error("cannot write threshold output " + output);
    std::fprintf(out, "Query, Counter, Position, Cross Correlation\n");

    for (int q = next_query(query_count); q < query_count; q = next_query(query_count))
    {
        const double start = MPI_Wtime();
        std::vector<ts_type> znorm, paa;
        std::vector<int> order;
        prepare_query(queries + static_cast<size_t>(q) * window, znorm, paa, order);

        std::vector<std::pair<file_position_type, float>> hits;
#pragma omp parallel num_threads(threads)
        {
            std::vector<std::pair<file_position_type, float>> local;
            int blocks = 0;
#pragma omp for schedule(dynamic, RANGE)
            for (long long id = 0; id < static_cast<long long>(subsequences); id++)
            {
                if (raw[id] == TREMOR_MISSING_VALUE || (lower_bounds && lower_bound(paa.data(), id) > bound))
                    continue;
                const float dist = l2_dist_znorm_subsequence(znorm.data(), raw.data(), id, window, means[id], stds[id],
                                                             bound, order.data(), &blocks);
                if (dist <= bound)
                    local.emplace_back(first_subsequence + id, dist);
            }
#pragma omp critical
            {
                hits.insert(hits.end(), local.begin(), local.end());
            }
        }

        // Same output as TREMOR: by position, merging detections closer than the merge offset (keep the first).
        std::sort(hits.begin(), hits.end());
        long long counter = 0;
        bool has_last = false;
        file_position_type last = 0;
        for (const auto &hit : hits)
        {
            if (has_last && merge_offset > 0 && hit.first < last + static_cast<file_position_type>(merge_offset + 1))
                continue;
            std::fprintf(out, "%d, %lld, %llu, %f\n", q, counter++, hit.first, ed_to_cc(hit.second, window));
            last = hit.first;
            has_last = true;
        }
        std::printf("[Node %d] query %d: %.6f s (%s)\n", rank, q, MPI_Wtime() - start, lower_bounds ? "sss" : "ss");
    }
    std::fclose(out);
}

void SkipSequentialScan::search_knn(const float *queries, int query_count, int k, const std::string &output)
{
    // With a merge offset, 2k-1 pairwise-separated windows bound the answer (as in TREMOR).
    const int bound_size = merge_offset > 0 ? 2 * k - 1 : k;
    std::vector<unsigned long long> positions;
    std::vector<float> distances;
    std::vector<int> counts(static_cast<size_t>(query_count));

    for (int q = next_query(query_count); q < query_count; q = next_query(query_count))
    {
        const double start = MPI_Wtime();
        std::vector<ts_type> znorm, paa;
        std::vector<int> order;
        prepare_query(queries + static_cast<size_t>(q) * window, znorm, paa, order);

        pqueue_bsf *best = pqueue_bsf_init(bound_size);
        std::vector<KnnCandidate> candidates;
        std::mutex lock;
#pragma omp parallel num_threads(threads)
        {
            int blocks = 0;
#pragma omp for schedule(dynamic, RANGE)
            for (long long id = 0; id < static_cast<long long>(subsequences); id++)
            {
                const float bound = best->knn[bound_size - 1];
                if (raw[id] == TREMOR_MISSING_VALUE || (lower_bounds && lower_bound(paa.data(), id) > bound))
                    continue;
                const float dist = l2_dist_znorm_subsequence(znorm.data(), raw.data(), id, window, means[id], stds[id],
                                                             bound, order.data(), &blocks);
                if (dist > bound)
                    continue;
                std::lock_guard<std::mutex> guard(lock);
                if (dist <= best->knn[bound_size - 1])
                    candidates.emplace_back(first_subsequence + id, dist);
                if (dist < best->knn[bound_size - 1])
                    pqueue_bsf_insert_offset(best, dist, first_subsequence + id, nullptr, merge_offset);
            }
        }

        // Keep the candidates within the final bound; rank 0 selects the answer below.
        const float bound = best->knn[bound_size - 1];
        for (const KnnCandidate &c : candidates)
        {
            if (c.second <= bound)
            {
                positions.push_back(c.first);
                distances.push_back(c.second);
                counts[q]++;
            }
        }
        pqueue_bsf_destroy(best);
        std::printf("[Node %d] query %d: %.6f s (%s)\n", rank, q, MPI_Wtime() - start, lower_bounds ? "sss" : "ss");
    }

    // Gather every rank's candidates on rank 0 and select each query's answer.
    const int local_total = static_cast<int>(positions.size());
    std::vector<int> all_counts(rank == 0 ? static_cast<size_t>(query_count) * ranks : 0);
    std::vector<int> totals(rank == 0 ? ranks : 0), displs(rank == 0 ? ranks : 0);
    MPI_Gather(counts.data(), query_count, MPI_INT, all_counts.data(), query_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_total, 1, MPI_INT, totals.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    size_t grand_total = 0;
    for (int r = 0; r < static_cast<int>(totals.size()); r++)
    {
        displs[r] = static_cast<int>(grand_total);
        grand_total += static_cast<size_t>(totals[r]);
    }
    std::vector<unsigned long long> all_positions(grand_total);
    std::vector<float> all_distances(grand_total);
    MPI_Gatherv(positions.data(), local_total, MPI_UNSIGNED_LONG_LONG, all_positions.data(), totals.data(),
                displs.data(), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gatherv(distances.data(), local_total, MPI_FLOAT, all_distances.data(), totals.data(), displs.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        return;

    std::vector<std::vector<KnnCandidate>> per_query(static_cast<size_t>(query_count));
    size_t at = 0;
    for (int r = 0; r < ranks; r++)
        for (int q = 0; q < query_count; q++)
            for (int j = 0; j < all_counts[static_cast<size_t>(r) * query_count + q]; j++, at++)
                per_query[q].emplace_back(all_positions[at], all_distances[at]);

    FILE *out = std::fopen(output.c_str(), "w");
    if (out == nullptr)
        throw std::runtime_error("cannot write kNN output " + output);
    std::fprintf(out, "tID, k, pos, corr\n");
    for (int q = 0; q < query_count; q++)
    {
        const std::vector<KnnCandidate> answer = select_knn_with_exclusion(per_query[q], k, merge_offset);
        for (int j = 0; j < k; j++)
        {
            const bool found = j < static_cast<int>(answer.size());
            std::fprintf(out, "%d, %d, %llu, %f\n", q, j + 1, found ? answer[j].first : 0ULL,
                         ed_to_cc(found ? answer[j].second : FLT_MAX, window));
        }
    }
    std::fclose(out);
}
