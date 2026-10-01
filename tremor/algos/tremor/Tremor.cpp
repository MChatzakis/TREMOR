#include "../htremor/Tremor.hpp"
#include "../htremor/indexing.hpp"
#include "../htremor/query_answering.hpp"
#include "../htremor/replication.hpp"
#include "../htremor/workstealing.hpp"
#include "../../isax/iSAXSearch.hpp"
#include "../../isax/iSAXIndex.hpp"
#include "../../isax/iSAXPqueue.hpp"
#include "../../isax/SAX.hpp"
#include "../../isax/iSAXTypes.hpp"
#include "../htremor/bsf_sharing.hpp"
#include "common.hpp"
#include "../../utils/SIMD.hpp"
#include "../../distance_computers/ZnormSearch.hpp"
#include "fft_fallback.hpp"
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <thread>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <set>
#include <cerrno>
#include <fstream>
#include <limits>
#include <mpi.h>
#include <unistd.h>

#ifndef CHECK_ALLOC
#define CHECK_ALLOC(ptr, rank)                                                                                   \
    do                                                                                                           \
    {                                                                                                            \
        if ((ptr) == nullptr)                                                                                    \
        {                                                                                                        \
            fprintf(stderr, "[Node %d] Error: Memory allocation failed in %s:%d\n", (rank), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                                                             \
        }                                                                                                        \
    } while (0)
#endif

// Debug flag for workstealing - can be used in all contexts
constexpr bool ENABLE_PRINTS_WORKSTEALING = false;

static RefinementMode g_refinement_mode = RefinementMode::Leaf;


// Resident memory of this process in bytes (Linux /proc).
static double resident_bytes()
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

// kNN with a merge offset (exclusion zone). The answer is canonical: sort
// all windows by (distance, position) and keep a window unless it lies
// within merge_offset of one already kept, until k are kept. It does not
// depend on the order in which windows are verified. The search keeps 2k-1
// windows pairwise merge_offset apart (pqueue_bsf_insert_offset); each of
// the first k-1 answers excludes at most 2 of them, so one remains for the
// k-th answer, and their largest distance is a safe pruning bound (for
// k = 1, the nearest-neighbor distance). Every verified window
// within that bound is a candidate; rank 0 selects the answer from the
// candidates of all ranks. Indexed by the query's position in the search.
static std::vector<std::vector<KnnCandidate>> g_knn_candidates;

// iSAX word of every owned subsequence in waveform order (filled by the index build), so that the
// adaptive fallback can skip subsequences by their lower bound like a skip-sequential scan.
static std::vector<sax_type> g_subsequence_sax;

// This rank's threshold output, opened by the first query and closed at the end of the search.
static FILE *g_threshold_output = nullptr;

static double g_scan_fallback_fraction = 0.01;
static std::atomic<long long> g_adaptive_leaf_queries{0};
static std::atomic<long long> g_adaptive_scan_queries{0};

void set_refinement_mode(RefinementMode mode)
{
    g_refinement_mode = mode;
}

void set_scan_fallback_fraction(double fraction)
{
    g_scan_fallback_fraction = fraction;
}

void adaptive_refinement_counts(long long *leaf_queries, long long *scan_queries)
{
    *leaf_queries = g_adaptive_leaf_queries.load();
    *scan_queries = g_adaptive_scan_queries.load();
}

static inline unsigned long long pack_float_for_mpi(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<unsigned long long>(bits);
}

static inline float unpack_float_from_mpi(unsigned long long value)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(value);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static idx_t tremor_float32_count_from_file(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Cannot open dataset file: " + filename);
    }

    const std::streamoff bytes = file.tellg();
    if (bytes < 0)
    {
        throw std::runtime_error("Cannot determine dataset file size: " + filename);
    }
    if (bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
    {
        throw std::runtime_error("Dataset file size is not a multiple of float32: " + filename);
    }

    return static_cast<idx_t>(bytes / static_cast<std::streamoff>(sizeof(float)));
}

static void tremor_read_float32_chunk(const std::string &filename,
                                      idx_t global_start,
                                      idx_t count,
                                      float *destination)
{
    if (count == 0)
    {
        return;
    }
    if (destination == nullptr)
    {
        throw std::runtime_error("Cannot read dataset chunk into a null destination");
    }
    if (global_start > static_cast<idx_t>(std::numeric_limits<std::streamoff>::max() / sizeof(float)))
    {
        throw std::overflow_error("Dataset chunk offset is too large");
    }
    if (count > static_cast<idx_t>(std::numeric_limits<size_t>::max() / sizeof(float)))
    {
        throw std::overflow_error("Dataset chunk is too large for this process");
    }

    const size_t byte_count = static_cast<size_t>(count) * sizeof(float);
    if (byte_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
    {
        throw std::overflow_error("Dataset chunk is too large for one read");
    }

    const std::streamoff byte_offset =
        static_cast<std::streamoff>(global_start) * static_cast<std::streamoff>(sizeof(float));

    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot open dataset file: " + filename);
    }

    file.seekg(byte_offset);
    if (!file)
    {
        throw std::runtime_error("Cannot seek dataset file: " + filename);
    }

    file.read(reinterpret_cast<char *>(destination), static_cast<std::streamsize>(byte_count));
    if (!file)
    {
        throw std::runtime_error("Failed to read dataset chunk from: " + filename);
    }
}

static inline float minidist_paa_to_raw_sax_safe(ts_type *paa, sax_type *sax, isax_index *index)
{
    if (index->settings->paa_segments == 16)
    {
        return minidist_paa_to_isax_rawa_SIMD(
            paa, sax,
            index->settings->max_sax_cardinalities,
            index->settings->sax_bit_cardinality,
            index->settings->sax_alphabet_cardinality,
            index->settings->paa_segments, MINVAL, MAXVAL,
            index->settings->mindist_sqrt);
    }

    return minidist_paa_to_isax(
        paa, sax,
        index->settings->max_sax_cardinalities,
        index->settings->sax_bit_cardinality,
        index->settings->sax_alphabet_cardinality,
        index->settings->paa_segments, MINVAL, MAXVAL,
        index->settings->mindist_sqrt);
}

static void tremor_write_knn_output_if_requested(const std::string &output_file,
                                                 int my_rank,
                                                 int q_num,
                                                 int topk,
                                                 const idx_t *I,
                                                 const float *D,
                                                 int subsequence_length)
{
    if (my_rank != 0 || output_file.empty() || I == nullptr || D == nullptr)
        return;

    FILE *of = std::fopen(output_file.c_str(), "w");
    if (of == nullptr)
    {
        std::fprintf(stderr, "[Node %d] Error opening output file %s\n", my_rank, output_file.c_str());
        std::exit(EXIT_FAILURE);
    }

    std::fprintf(of, "tID, k, pos, corr\n");
    for (int q = 0; q < q_num; q++)
    {
        for (int j = 0; j < topk; j++)
        {
            const size_t out = static_cast<size_t>(q) * static_cast<size_t>(topk) + static_cast<size_t>(j);
            std::fprintf(of, "%d, %d, %llu, %f\n",
                         q,
                         j + 1,
                         static_cast<unsigned long long>(I[out]),
                         ed_to_cc(D[out], subsequence_length));
        }
    }

    std::fclose(of);
}

static void tremor_prepare_threshold_output_file(const std::string &output_file, int my_rank)
{
    if (output_file.empty())
        return;

    char node_output_name[512];
    std::snprintf(node_output_name, sizeof(node_output_name), "%s_%d.csv", output_file.c_str(), my_rank);
    FILE *of = std::fopen(node_output_name, "w");
    if (of == nullptr)
    {
        std::fprintf(stderr, "[Node %d] Error: Cannot open threshold output file %s\n", my_rank, node_output_name);
        std::exit(EXIT_FAILURE);
    }
    std::fprintf(of, "Query, Counter, Position, Cross Correlation\n");
    std::fclose(of);
}

// kNN with a merge offset: gather every rank's candidates on rank 0 and
// select the canonical answer of each query (see g_knn_candidates).
// Missing answers are written as (0, FLT_MAX), as without a merge offset.
static void tremor_knn_exclusion_answers(int my_rank, int comm_sz, int q_num, int topk, int merge_offset,
                                         const TremorQuery *queries, idx_t *I, float *D)
{
    // Local candidates by original query id.
    std::vector<int> counts(static_cast<size_t>(q_num), 0);
    std::vector<const std::vector<KnnCandidate> *> by_id(static_cast<size_t>(q_num), nullptr);
    for (int i = 0; i < q_num && i < static_cast<int>(g_knn_candidates.size()); i++)
        by_id[static_cast<size_t>(queries[i].id)] = &g_knn_candidates[static_cast<size_t>(i)];
    std::vector<unsigned long long> positions;
    std::vector<float> distances;
    for (int q = 0; q < q_num; q++)
    {
        if (by_id[static_cast<size_t>(q)] == nullptr)
            continue;
        for (const KnnCandidate &c : *by_id[static_cast<size_t>(q)])
        {
            positions.push_back(static_cast<unsigned long long>(c.first));
            distances.push_back(c.second);
        }
        counts[static_cast<size_t>(q)] = static_cast<int>(by_id[static_cast<size_t>(q)]->size());
    }

    const int local_total = static_cast<int>(positions.size());
    std::vector<int> all_counts(my_rank == 0 ? static_cast<size_t>(q_num) * comm_sz : 0);
    std::vector<int> totals(my_rank == 0 ? comm_sz : 0), displs(my_rank == 0 ? comm_sz : 0);
    MPI_Gather(counts.data(), q_num, MPI_INT, all_counts.data(), q_num, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_total, 1, MPI_INT, totals.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    size_t grand_total = 0;
    if (my_rank == 0)
    {
        for (int r = 0; r < comm_sz; r++)
        {
            displs[static_cast<size_t>(r)] = static_cast<int>(grand_total);
            grand_total += static_cast<size_t>(totals[static_cast<size_t>(r)]);
        }
    }
    std::vector<unsigned long long> all_positions(grand_total);
    std::vector<float> all_distances(grand_total);
    MPI_Gatherv(positions.data(), local_total, MPI_UNSIGNED_LONG_LONG, all_positions.data(), totals.data(),
                displs.data(), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gatherv(distances.data(), local_total, MPI_FLOAT, all_distances.data(), totals.data(),
                displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        std::vector<std::vector<KnnCandidate>> per_query(static_cast<size_t>(q_num));
        size_t at = 0;
        for (int r = 0; r < comm_sz; r++)
        {
            for (int q = 0; q < q_num; q++)
            {
                const int n = all_counts[static_cast<size_t>(r) * q_num + q];
                for (int j = 0; j < n; j++, at++)
                    per_query[static_cast<size_t>(q)].emplace_back(all_positions[at], all_distances[at]);
            }
        }
        for (int q = 0; q < q_num; q++)
        {
            const std::vector<KnnCandidate> answer = select_knn_with_exclusion(per_query[static_cast<size_t>(q)], topk, merge_offset);
            for (int j = 0; j < topk; j++)
            {
                const size_t out = static_cast<size_t>(q) * topk + j;
                I[out] = j < static_cast<int>(answer.size()) ? static_cast<idx_t>(answer[static_cast<size_t>(j)].first) : 0;
                D[out] = j < static_cast<int>(answer.size()) ? answer[static_cast<size_t>(j)].second : FLT_MAX;
            }
        }
    }
}

static void tremor_merge_knn_results_mpi(int my_rank, int comm_sz, int q_num, int topk, idx_t *I, float *D, int merge_offset)
{
    if (comm_sz <= 1)
        return;
    const size_t per_rank = static_cast<size_t>(q_num) * static_cast<size_t>(topk);
    idx_t *all_I = nullptr;
    float *all_D = nullptr;
    if (my_rank == 0)
    {
        all_I = static_cast<idx_t *>(std::malloc(per_rank * static_cast<size_t>(comm_sz) * sizeof(idx_t)));
        all_D = static_cast<float *>(std::malloc(per_rank * static_cast<size_t>(comm_sz) * sizeof(float)));
        CHECK_ALLOC(all_I, my_rank);
        CHECK_ALLOC(all_D, my_rank);
    }
    MPI_Gather(I, static_cast<int>(per_rank), MPI_UNSIGNED_LONG_LONG,
               all_I, static_cast<int>(per_rank), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(D, static_cast<int>(per_rank), MPI_FLOAT,
               all_D, static_cast<int>(per_rank), MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        for (int q = 0; q < q_num; q++)
        {
            pqueue_bsf *final_result_pq = pqueue_bsf_init(topk);
            CHECK_ALLOC(final_result_pq, my_rank);
            for (int r = 0; r < comm_sz; r++)
            {
                for (int j = 0; j < topk; j++)
                {
                    size_t idx = static_cast<size_t>(r) * per_rank + static_cast<size_t>(q) * static_cast<size_t>(topk) + static_cast<size_t>(j);
                    idx_t pos = all_I[idx];
                    float dist = all_D[idx];

                    if (dist >= FLT_MAX * 0.99f)
                        continue;

                    if (pos == static_cast<idx_t>(static_cast<long>(-1)))
                        continue;
                    pqueue_bsf_insert_offset(final_result_pq, dist, pos, nullptr, merge_offset);
                }
            }
            idx_t *out_I = I + q * topk;
            float *out_D = D + q * topk;

            for (int j = 0; j < topk; j++)
            {
                out_I[j] = final_result_pq->position[j] >= 0
                               ? static_cast<idx_t>(final_result_pq->position[j])
                               : 0;
                out_D[j] = final_result_pq->knn[j];
            }
            pqueue_bsf_destroy(final_result_pq);
        }
        std::free(all_I);
        std::free(all_D);
    }

    MPI_Bcast(I, static_cast<int>(per_rank), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(D, static_cast<int>(per_rank), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

TremorQuery *load_queries_from_buffer(const float *query_buf, int q_num, isax_index *index, int my_rank)
{
    const int ts_len = index->settings->timeseries_size;
    TremorQuery *queries = (TremorQuery *)malloc(sizeof(TremorQuery) * static_cast<size_t>(q_num));
    CHECK_ALLOC(queries, my_rank);

    for (int i = 0; i < q_num; i++)
    {
        queries[i].id = i;
        queries[i].initial_estimation = 0.0;
        queries[i].initial_pq_bsfs = nullptr;
        queries[i].paa = nullptr;

        queries[i].query = (ts_type *)malloc(sizeof(ts_type) * static_cast<size_t>(ts_len));
        CHECK_ALLOC(queries[i].query, my_rank);

        std::memcpy(queries[i].query, query_buf + static_cast<size_t>(i) * static_cast<size_t>(ts_len),
                    sizeof(ts_type) * static_cast<size_t>(ts_len));
    }
    return queries;
}

void free_queries(TremorQuery *queries, int q_num)
{
    if (!queries)
        return;
    for (int i = 0; i < q_num; i++)
    {
        if (queries[i].paa)
        {
            free(queries[i].paa);
            queries[i].paa = nullptr;
        }
        if (queries[i].initial_pq_bsfs)
        {
            pqueue_bsf_destroy(queries[i].initial_pq_bsfs);
            queries[i].initial_pq_bsfs = nullptr;
        }
        free(queries[i].query);
        queries[i].query = nullptr;
    }
    free(queries);
}

double predict_exec_time(float bsf, const char *dataset_type)
{
    if (strcmp(dataset_type, "default") == 0)
    {
        return static_cast<double>(bsf);
    }
    else if (strcmp(dataset_type, "seismic") == 0)
    {
        return static_cast<double>(bsf);
    }
    else if (strcmp(dataset_type, "astro") == 0)
    {
        return static_cast<double>(bsf);
    }
    else if (strcmp(dataset_type, "deep") == 0)
    {
        return static_cast<double>(bsf);
    }
    else if (strcmp(dataset_type, "sift") == 0)
    {
        return static_cast<double>(bsf);
    }
    else if (strcmp(dataset_type, "t2i") == 0)
    {
        return static_cast<double>(bsf);
    }
    else if (strcmp(dataset_type, "random") == 0)
    {
        return static_cast<double>(bsf);
    }
    else
    {
        printf("ERROR: Dataset type %s does not have an associated query time prediction function.\n", dataset_type);
        std::exit(EXIT_FAILURE);
    }

    return static_cast<double>(bsf);
}

// Prefetch distances (in records) used by the leaf scan.
constexpr int RECORD_PREFETCH_FAR = 16;
constexpr int RECORD_PREFETCH_NEAR = 6;
#ifndef TREMOR_PREFETCH_MAX_BLOCKS
#define TREMOR_PREFETCH_MAX_BLOCKS 8
#endif
constexpr int RECORD_PREFETCH_MAX_BLOCKS = TREMOR_PREFETCH_MAX_BLOCKS;

[[maybe_unused]] static void rerank_pq_exact_l2(pqueue_bsf *pq, ts_type *query, float *rawfile, int dim)
{
    if (!pq || !rawfile || !query)
        return;

    const int k = pq->k;
    std::vector<std::pair<float, file_position_type>> cand;
    cand.reserve(static_cast<size_t>(k));

    for (int i = 0; i < k; i++)
    {
        file_position_type pos = static_cast<file_position_type>(pq->position[i]);
        if (pos < 0)
            continue;

        const ts_type *base = rawfile + static_cast<size_t>(pos) * static_cast<size_t>(dim);
        float dist = 0.0f;
        for (int d = 0; d < dim; d++)
        {
            float diff = static_cast<float>(query[d] - base[d]);
            dist += diff * diff;
        }
        cand.emplace_back(dist, pos);
    }

    if (cand.empty())
        return;

    std::sort(cand.begin(), cand.end(), [](const auto &a, const auto &b)
              {
            if (a.first != b.first)
                return a.first < b.first;
            return a.second < b.second; });

    std::vector<std::pair<float, file_position_type>> uniq;
    uniq.reserve(cand.size());
    file_position_type last_pos = static_cast<file_position_type>(-1);
    for (const auto &p : cand)
    {
        if (p.second != last_pos)
        {
            uniq.push_back(p);
            last_pos = p.second;
        }
    }

    for (int i = 0; i < k; i++)
    {
        pq->knn[i] = FLT_MAX;
        pq->position[i] = -1;
    }

    int copy_count = std::min(k, static_cast<int>(uniq.size()));
    for (int i = 0; i < copy_count; i++)
    {
        pq->knn[i] = uniq[static_cast<size_t>(i)].first;
        pq->position[i] = static_cast<long>(uniq[static_cast<size_t>(i)].second);
    }

    if (copy_count > 0 && copy_count < k)
    {
        float pad_dist = pq->knn[copy_count - 1];
        long pad_pos = pq->position[copy_count - 1];
        for (int i = copy_count; i < k; i++)
        {
            pq->knn[i] = pad_dist;
            pq->position[i] = pad_pos;
        }
    }
}

int cmp_query(const void *a, const void *b)
{
    const TremorQuery *query_a = static_cast<const TremorQuery *>(a);
    const TremorQuery *query_b = static_cast<const TremorQuery *>(b);

    if (query_a->initial_estimation < query_b->initial_estimation)
    {
        return -1;
    }
    else if (query_a->initial_estimation > query_b->initial_estimation)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void tremor_preprocess_and_sort_queries(Tremor *tremor, TremorQuery *queries, int q_num, bool apply_sort)
{
    const int MASTER = 0;

    isax_index *index = tremor->index;
    const char *dataset_type = tremor->dataset_type.c_str();
    int comm_sz = tremor->comm_sz;
    int my_rank = tremor->my_rank;
    int topk = tremor->top_k;
    BsfSharingData &bsf_sharing_data = tremor->bsf_sharing_data;
    bool verbose = tremor->verbose;
    float *rawfile = tremor->rawfile;

    for (int i = 0; i < q_num; i++)
    {
        znormalize_inplace(queries[i].query, index->settings->timeseries_size);

        queries[i].paa = (ts_type *)malloc(sizeof(ts_type) * static_cast<size_t>(index->settings->paa_segments));
        CHECK_ALLOC(queries[i].paa, my_rank);

        paa_from_ts(queries[i].query, queries[i].paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

        {
            const bool threshold_mode = (tremor->mode == 0);
            if (threshold_mode)
            {
                const float threshold_ed = cc_to_ed(tremor->corr_threshold, index->settings->timeseries_size);
                queries[i].initial_pq_bsfs = pqueue_bsf_init_from_val(topk, threshold_ed, static_cast<file_position_type>(-1));
            }
            else
            {
                queries[i].initial_pq_bsfs = pqueue_bsf_init(topk);
            }
        }
    }

    if (!apply_sort)
    {
        return;
    }

    {
        float *all_bsfs = (float *)malloc(sizeof(float) * static_cast<size_t>(q_num) * static_cast<size_t>(topk));
        CHECK_ALLOC(all_bsfs, my_rank);

        file_position_type *all_pos = (file_position_type *)malloc(sizeof(file_position_type) * static_cast<size_t>(q_num) * static_cast<size_t>(topk));
        CHECK_ALLOC(all_pos, my_rank);

        for (int i = 0; i < q_num; i++)
        {
            for (int j = 0; j < topk; j++)
            {
                all_bsfs[i * topk + j] = queries[i].initial_pq_bsfs->knn[j];
                all_pos[i * topk + j] = queries[i].initial_pq_bsfs->position[j];
            }
        }

        if (comm_sz > 1 && static_cast<int>(bsf_sharing_data.communicators.size()) > my_rank)
        {
            MPI_Request local_send_requests[2];
            MPI_Ibcast(all_bsfs, q_num * topk, MPI_FLOAT, my_rank, bsf_sharing_data.communicators[my_rank], &local_send_requests[0]);
            MPI_Ibcast(all_pos, q_num * topk, MPI_UNSIGNED_LONG_LONG, my_rank, bsf_sharing_data.communicators[my_rank], &local_send_requests[1]);

            float *all_bsfs_recv = (float *)malloc(sizeof(float) * static_cast<size_t>(q_num) * static_cast<size_t>(topk));
            CHECK_ALLOC(all_bsfs_recv, my_rank);

            file_position_type *all_pos_recv = (file_position_type *)malloc(sizeof(file_position_type) * static_cast<size_t>(q_num) * static_cast<size_t>(topk));
            CHECK_ALLOC(all_pos_recv, my_rank);

            for (int i = 0; i < comm_sz; i++)
            {
                if (i == my_rank)
                    continue;

                MPI_Request recv_req;
                MPI_Ibcast(all_bsfs_recv, q_num * topk, MPI_FLOAT, i, bsf_sharing_data.communicators[i], &recv_req);
                MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

                MPI_Ibcast(all_pos_recv, q_num * topk, MPI_UNSIGNED_LONG_LONG, i, bsf_sharing_data.communicators[i], &recv_req);
                MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

                for (int j = 0; j < q_num; j++)
                {
                    for (int kk = 0; kk < topk; kk++)
                    {
                        float current_k_bsf = all_bsfs_recv[j * topk + kk];
                        file_position_type current_k_pos = all_pos_recv[j * topk + kk];

                        if (current_k_bsf < queries[j].initial_pq_bsfs->knn[topk - 1])
                        {
                            pqueue_bsf_insert(queries[j].initial_pq_bsfs, current_k_bsf, static_cast<long int>(current_k_pos), nullptr);
                        }
                    }
                }
            }

            bsf_sharing_data.shared_bsfs[my_rank].bsf = queries[0].initial_pq_bsfs->knn[topk - 1];
            bsf_sharing_data.shared_bsfs[my_rank].position = queries[0].initial_pq_bsfs->position[topk - 1];
            bsf_sharing_data.shared_bsfs[my_rank].q_num = 0;

            MPI_Waitall(2, local_send_requests, MPI_STATUSES_IGNORE);

            free(all_bsfs_recv);
            free(all_pos_recv);

            for (int process_rank = 0; process_rank < comm_sz; process_rank++)
            {
                if (process_rank == my_rank)
                    continue;
                if (static_cast<int>(bsf_sharing_data.requests.size()) > process_rank)
                    MPI_Ibcast(&bsf_sharing_data.shared_bsfs[process_rank], 1, bsf_msg_type, process_rank, bsf_sharing_data.communicators[process_rank], &bsf_sharing_data.requests[process_rank]);
            }
        }
        free(all_bsfs);
        free(all_pos);
    }

    for (int i = 0; i < q_num; i++)
    {
        float k_th_bsf = queries[i].initial_pq_bsfs->knn[topk - 1];
        queries[i].initial_estimation = predict_exec_time(k_th_bsf, dataset_type);
    }

    // Only sort queries when running on a single rank. With multiple
    // ranks the sort key is derived from BSFs that may be tied across
    // ranks, so qsort can produce different orders on different ranks.
    // The dynamic-scheduling worker path indexes queries by the order
    // received from the coordinator, so a divergent order would route
    // results to the wrong query slot.
    if (apply_sort && comm_sz <= 1)
    {
        qsort(queries, static_cast<size_t>(q_num), sizeof(TremorQuery), cmp_query);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == MASTER && verbose)
    {
        printf("[Node %d]: Initial BSFs exchanged. Proceeding to QA.\n", my_rank);
    }
}

constexpr int DISTRIBUTED_QUERIES_SEND_QUERY = 800;
constexpr int DISTRIBUTED_QUERIES_REQUEST_QUERY = 801;
constexpr int DYNAMIC_TERMINATION_MESSAGE = -1;

static void shuffle_int_array(int *arr, int n)
{
    static std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
    for (int i = n - 1; i > 0; i--)
    {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
}

void send_initial_queries_module_coordinator_async_chatzakis(int *q_loaded, int my_rank, int comm_sz,
                                                             int distributed_queries_initial_burst,
                                                             int **process_buffer_initial,
                                                             int *rec_message, MPI_Request *request, MPI_Request *send_request,
                                                             int q_num, int *termination_message_id,
                                                             ReplicationData *replication_data)
{
    (void)q_num;
    (void)termination_message_id;
    int coordinator_of_current_group_rank = rep_find_coordinator_node_rank(*replication_data, my_rank);
    int repgroup_nodes = rep_get_repgroup_nodes(*replication_data, my_rank);

    for (int i = 0; i < distributed_queries_initial_burst; i++)
    {
        for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            process_buffer_initial[rank][i] = (*q_loaded)++;
        }
    }

    for (int i = 0; i < distributed_queries_initial_burst; i++)
    {
        for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            MPI_Isend(&process_buffer_initial[rank][i], 1, MPI_INT, rank, DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, &send_request[rank]);
        }
    }

    for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
    {
        MPI_Irecv(rec_message, 1, MPI_INT, rank, DISTRIBUTED_QUERIES_REQUEST_QUERY, MPI_COMM_WORLD, &request[rank]);
    }
}

int send_queries_module_coordinator_async_chatzakis(int *q_loaded, int q_num, int *process_buffer, MPI_Request *request, int *rec_message,
                                                    MPI_Request *send_request, int *termination_message_id,
                                                    ReplicationData *replication_data, int my_rank, int comm_sz,
                                                    bool verbose)
{
    int ready;
    int coordinator_of_current_group_rank = rep_find_coordinator_node_rank(*replication_data, my_rank);
    int repgroup_nodes = rep_get_repgroup_nodes(*replication_data, my_rank);

    bool termination_message_sent = false;
    if (termination_message_sent)
    {
        return 0;
    }

    bool *node_requested = (bool *)malloc(sizeof(bool) * static_cast<size_t>(comm_sz));
    CHECK_ALLOC(node_requested, my_rank);

    for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
    {
        MPI_Test(&request[rank], &ready, MPI_STATUS_IGNORE);

        if (ready)
        {
            MPI_Wait(&send_request[rank], MPI_STATUS_IGNORE);

            process_buffer[rank] = (*q_loaded)++;
            node_requested[rank] = true;

            MPI_Irecv(rec_message, 1, MPI_INT, rank, DISTRIBUTED_QUERIES_REQUEST_QUERY, MPI_COMM_WORLD, &request[rank]);
        }
        else
        {
            node_requested[rank] = false;
        }
    }

    for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
    {
        if (node_requested[rank])
        {
            MPI_Isend(&process_buffer[rank], 1, MPI_INT, rank, DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, &send_request[rank]);
        }
    }

    free(node_requested);

    if ((*q_loaded) >= q_num)
    {
        for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            if (verbose)
                printf("[Node %d]: Sending to node %d a termination message\n", my_rank, rank);

            MPI_Isend(termination_message_id, 1, MPI_INT, rank, DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, &send_request[rank]);
        }

        termination_message_sent = true;
        return 0;
    }

    return 1;
}

void tremor_perform_workstealing(Tremor *tremor, TremorQuery *queries, NodeList nodelist,
                                 ws_func_type ws_func, double (*estimation_func)(double),
                                 query_result *results, std::vector<BsfMessage> *shared_bsf_results)
{
    int my_rank = tremor->my_rank;
    int comm_sz = tremor->comm_sz;
    int query_threads = tremor->query_threads;
    isax_index *index = tremor->index;
    ReplicationData *replication_data = &tremor->replication_data;
    WorkstealingData *workstealing_data = &tremor->workstealing_data;
    bool verbose = tremor->verbose;
    const float minimum_distance = FLT_MAX;
    int topk = tremor->top_k;

    ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * static_cast<size_t>(index->settings->paa_segments));
    CHECK_ALLOC(paa, my_rank);

    int term_message = DYNAMIC_TERMINATION_MESSAGE;
    int recv_message = 0;
    int ready = 0;
    std::vector<MPI_Request> send_request(static_cast<size_t>(comm_sz));
    std::vector<MPI_Request> recv_request(static_cast<size_t>(comm_sz));

    int repgroup_nodes = rep_get_repgroup_nodes(*replication_data, my_rank);
    int coordinator_of_current_group_rank = rep_find_coordinator_node_rank(*replication_data, my_rank);

    std::vector<int> nodes_of_group(static_cast<size_t>(repgroup_nodes));
    int in = 0;
    for (int rank = coordinator_of_current_group_rank; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
    {
        nodes_of_group[static_cast<size_t>(in++)] = rank;
    }

    shuffle_int_array(nodes_of_group.data(), repgroup_nodes);

    for (int i = 0; i < repgroup_nodes; i++)
    {
        int rank = nodes_of_group[static_cast<size_t>(i)];
        if (rank != my_rank)
        {
            if (ENABLE_PRINTS_WORKSTEALING && verbose)
                printf("[WORKSTEALING - Node %d]: Sending to node %d that it has finished working.\n", my_rank, rank);

            MPI_Isend(&term_message, 1, MPI_INT, rank, WORKSTEALING_QUERY_ANSWERING_COMPLETION, MPI_COMM_WORLD, &send_request[static_cast<size_t>(rank)]);
        }
    }

    for (int i = 0; i < repgroup_nodes; i++)
    {
        int rank = nodes_of_group[static_cast<size_t>(i)];
        if (rank != my_rank)
        {
            MPI_Irecv(&recv_message, 1, MPI_INT, rank, WORKSTEALING_QUERY_ANSWERING_COMPLETION, MPI_COMM_WORLD, &recv_request[static_cast<size_t>(rank)]);
        }
    }

    std::vector<bool> working_nodes(static_cast<size_t>(repgroup_nodes), true);
    int finished_nodes = 1;

    while (finished_nodes < repgroup_nodes)
    {
        for (int i = 0; i < repgroup_nodes; i++)
        {
            int rank = nodes_of_group[static_cast<size_t>(i)];

            if (rank != my_rank && working_nodes[static_cast<size_t>(i)])
            {
                MPI_Test(&recv_request[static_cast<size_t>(rank)], &ready, MPI_STATUS_IGNORE);

                if (ready)
                {
                    if (ENABLE_PRINTS_WORKSTEALING && verbose)
                        printf("[WORKSTEALING HELPER - Node %d]: Found that node %d has already finished his work.\n", my_rank, rank);
                    working_nodes[static_cast<size_t>(i)] = false;
                    finished_nodes++;
                }
                else
                {
                    if (ENABLE_PRINTS_WORKSTEALING && verbose)
                        printf("[WORKSTEALING HELPER - Node %d] - Found that node %d is still working. Sending a steal request.\n", my_rank, rank);

                    MPI_Isend(&term_message, 1, MPI_INT, rank, WORKSTEALING_INFORM_AVAILABILITY, MPI_COMM_WORLD, &send_request[static_cast<size_t>(rank)]);

                    int data_size = 3 + workstealing_data->items_to_send;
                    std::vector<unsigned long long> datas(static_cast<size_t>(data_size));

                    MPI_Request data_req;
                    MPI_Irecv(datas.data(), data_size, MPI_UNSIGNED_LONG_LONG, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD, &data_req);

                    int req_ready = 0;
                    MPI_Test(&data_req, &req_ready, MPI_STATUS_IGNORE);
                    while (!req_ready)
                    {
                        MPI_Test(&data_req, &req_ready, MPI_STATUS_IGNORE);
                        MPI_Test(&recv_request[static_cast<size_t>(rank)], &ready, MPI_STATUS_IGNORE);
                        if (ready)
                        {
                            if (ENABLE_PRINTS_WORKSTEALING && verbose)
                                printf("[WORKSTEALING - Node %d]: Found that node %d has already finished his work (While waiting to receive stolen work).\n", my_rank, rank);
                            working_nodes[static_cast<size_t>(i)] = false;
                            finished_nodes++;
                            req_ready = 1;
                        }
                    }

                    if (working_nodes[static_cast<size_t>(i)] == false)
                        break;

                    int query_num = (datas[0] == static_cast<unsigned long long>(-1))
                                        ? -1
                                        : static_cast<int>(datas[0]);
                    float bsf = unpack_float_from_mpi(datas[1]);
                    file_position_type bsf_position = static_cast<file_position_type>(datas[2]);

                    if (ENABLE_PRINTS_WORKSTEALING && verbose)
                    {
                        printf("[WORKSTEALING HELPER - Node %d]: Received message from node %d : [%d, %f, %llu, ", my_rank, rank, query_num, bsf, (unsigned long long)bsf_position);
                        for (int j = 0; j < workstealing_data->items_to_send; j++)
                            printf("%d ", static_cast<int>(datas[3 + static_cast<size_t>(j)]));
                        printf("]\n");
                    }

                    if (query_num >= 0)
                    {
                        std::vector<int> batches_to_create(static_cast<size_t>(workstealing_data->items_to_send));
                        for (int j = 0; j < workstealing_data->items_to_send; j++)
                            batches_to_create[static_cast<size_t>(j)] = static_cast<int>(datas[3 + static_cast<size_t>(j)]);

                        NodeList final_node_list = nodelist;
                        isax_node *original_lca_node = nullptr;

                        WsSearchFunctionParams ws_params;
                        ws_params.bsf_pos = bsf_position;
                        ws_params.ts = queries[query_num].query;
                        ws_params.paa = queries[query_num].paa;
                        ws_params.query_id = query_num;
                        ws_params.index = index;
                        ws_params.minimum_distance = minimum_distance;
                        ws_params.nodelist = &final_node_list;
                        ws_params.bsf = bsf;
                        ws_params.estimation_func = estimation_func;
                        ws_params.batch_ids = batches_to_create.data();
                        ws_params.shared_bsf_results = shared_bsf_results;
                        ws_params.comm_data = nullptr;
                        ws_params.lca_node = original_lca_node;
                        ws_params.k = topk;
                        ws_params.precomputed_bsfs = queries[query_num].initial_pq_bsfs;
                        ws_params.my_rank = my_rank;
                        ws_params.comm_sz = comm_sz;
                        ws_params.query_threads = query_threads;
                        ws_params.verbose = verbose;
                        ws_params.rawfile = tremor->rawfile;
                        ws_params.replication_data = replication_data;
                        ws_params.output_file = tremor->output_file;
                        ws_params.corr_threshold = tremor->corr_threshold;
                        ws_params.bsf_sharing_data = &tremor->bsf_sharing_data;
                        ws_params.workstealing_data = workstealing_data;
                        ws_params.pq_th_div_factor = tremor->pq_th_div_factor;
                        ws_params.merge_offset = tremor->merge_offset;
                        ws_params.query_counter = query_num;
                        ws_params.warp_window = 0;
                        ws_params.paaU = nullptr;
                        ws_params.paaL = nullptr;
                        ws_params.local_chunk_start = tremor->local_chunk_start;
                        ws_params.local_chunk_length = tremor->local_chunk_length;
                        ws_params.local_owned_subseq_start = tremor->local_owned_subseq_start;
                        ws_params.local_owned_subseq_count = tremor->local_owned_subseq_count;
                        ws_params.subsequence_means = tremor->subsequence_means;
                        ws_params.subsequence_stds = tremor->subsequence_stds;
                        ws_params.subsequence_length = tremor->time_series_size;
                        ws_params.threshold_mode = (tremor->mode == 0);
                        ws_params.threshold_ed = cc_to_ed(tremor->corr_threshold, tremor->time_series_size);
                        ws_params.threshold_output_file = nullptr;
                        ws_params.threshold_output_lock = nullptr;
                        ws_params.threshold_match_counter = nullptr;
                        ws_params.threshold_hits = nullptr;
                        ws_params.threshold_hits_lock = nullptr;

                        query_result result = ws_func(ws_params);

                        if (results[query_num].pq_bsf != nullptr)
                        {
                            for (int knni = 0; knni < result.pq_bsf->k; knni++)
                            {
                                if (result.pq_bsf->knn[knni] < results[query_num].pq_bsf->knn[results[query_num].pq_bsf->k - 1])
                                {
                                    pqueue_bsf_insert_offset(results[query_num].pq_bsf,
                                                             result.pq_bsf->knn[knni],
                                                             static_cast<file_position_type>(result.pq_bsf->position[knni]),
                                                             nullptr,
                                                             tremor->merge_offset);
                                }
                            }
                            pqueue_bsf_destroy(result.pq_bsf);
                        }
                        else
                        {
                            results[query_num] = result;
                        }

                        if (ENABLE_PRINTS_WORKSTEALING && verbose)
                        {
                            printf("[Node: %d]: Stolen query: %d (id %d), result: %f\n", my_rank, query_num, queries[query_num].id,
                                   results[query_num].pq_bsf != nullptr ? results[query_num].pq_bsf->knn[0] : 0.0f);
                        }
                    }
                    else
                    {
                        working_nodes[static_cast<size_t>(i)] = false;
                        finished_nodes++;
                    }
                }
            }
        }
    }

    free(paa);
}

NodeList initialize_node_list(isax_index *index, int my_rank)
{
    NodeList nodelist;
    const int capacity = (1 << index->settings->paa_segments);
    nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * static_cast<size_t>(capacity));
    CHECK_ALLOC(nodelist.nlist, my_rank);
    if (!nodelist.nlist)
    {
        printf("Error: nodelist.nlist is NULL!\n");
        std::exit(EXIT_FAILURE);
    }
    nodelist.node_amount = 0;
    nodelist.data_amount = 0;
    nodelist.rawfile = nullptr;

    parallel_first_buffer_layer_ekosmas *fbl = (parallel_first_buffer_layer_ekosmas *)(index->fbl);
    if (!fbl)
    {
        return nodelist;
    }

    for (int j = 0; j < fbl->number_of_buffers; j++)
    {
        parallel_fbl_soft_buffer_ekosmas *current_fbl_node = &fbl->soft_buffers[j];
        if (!current_fbl_node->initialized)
        {
            continue;
        }

        nodelist.nlist[nodelist.node_amount] = current_fbl_node->node;
        if (!current_fbl_node->node)
        {
            printf("Error: node is NULL! (buffer %d)\n", j);
            std::fflush(stdout);
            std::exit(EXIT_FAILURE);
        }
        nodelist.node_amount++;
    }

    return nodelist;
}

int estimate_th(double x, double (*estimation_func)(double))
{
    if (estimation_func == nullptr)
    {
        printf("estimate_th: PQ TH estimation function is NULL\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(estimation_func(x));
}

static int pq_comparator(const void *a, const void *b)
{
    pqueue_t *p = *(pqueue_t **)a;
    pqueue_t *q = *(pqueue_t **)b;
    query_result *ra = static_cast<query_result *>(pqueue_peek(p));
    query_result *rb = static_cast<query_result *>(pqueue_peek(q));
    if (ra == nullptr)
        return 1;
    if (rb == nullptr)
        return -1;
    if (ra->distance < rb->distance)
        return -1;
    if (ra->distance > rb->distance)
        return 1;
    return 0;
}

void generate_pqs_of_rs_batch(isax_node *subtree_node, SubtreeBatch *batch, float bsf_distance, ts_type *paa, isax_index *index)
{
    if (subtree_node == nullptr || subtree_node->isax_values == nullptr || subtree_node->isax_cardinalities == nullptr)
        return;

    float distance = minidist_paa_to_isax(paa, subtree_node->isax_values, subtree_node->isax_cardinalities,
                                          index->settings->sax_bit_cardinality, index->settings->sax_alphabet_cardinality,
                                          index->settings->paa_segments, MINVAL, MAXVAL, index->settings->mindist_sqrt);

    if (distance >= bsf_distance)
        return;

    if (subtree_node->is_leaf)
    {
        query_result *mindist_result = static_cast<query_result *>(std::malloc(sizeof(query_result)));
        if (mindist_result == nullptr)
        {
            printf("MEMORY ERROR: query_result allocation failed. Exiting...\n");
            std::exit(EXIT_FAILURE);
        }
        mindist_result->node = subtree_node;
        mindist_result->distance = distance;

        pthread_mutex_lock(&batch->pq_insert_lock);

        if (batch->pq_amount >= MAX_PQs_WORKSTEALING)
        {
            printf("MEMORY ERROR: Priority queues per batch exceeded the limit of %d. Increase the limit. Exiting...\n", MAX_PQs_WORKSTEALING);
            pthread_mutex_unlock(&batch->pq_insert_lock);
            std::free(mindist_result);
            std::exit(EXIT_FAILURE);
        }

        if (batch->pq[batch->pq_amount] == nullptr)
        {
            batch->pq[batch->pq_amount] = pqueue_init(static_cast<size_t>(batch->pq_th), cmp_pri, get_pri, set_pri, get_pos, set_pos);
            if (batch->pq[batch->pq_amount] == nullptr)
            {
                printf("MEMORY ERROR: Priority queue allocation failed. Exiting...\n");
                pthread_mutex_unlock(&batch->pq_insert_lock);
                std::free(mindist_result);
                std::exit(EXIT_FAILURE);
            }
            batch->pq[batch->pq_amount]->is_stolen = 0;
            batch->pq[batch->pq_amount]->is_processed = 0;
            batch->pq[batch->pq_amount]->batch_id = batch->id;
            batch->pq[batch->pq_amount]->starting_node = subtree_node;
            batch->pq[batch->pq_amount]->ending_node = nullptr;
            batch->pq[batch->pq_amount]->lca_node = nullptr;
        }

        if (pqueue_insert(batch->pq[batch->pq_amount], mindist_result) != 0)
        {
            printf("MEMORY ERROR: Priority queue insertion failed. Exiting...\n");
            pthread_mutex_unlock(&batch->pq_insert_lock);
            std::free(mindist_result);
            std::exit(EXIT_FAILURE);
        }

        batch->pq[batch->pq_amount]->ending_node = subtree_node;

        if (pqueue_size(batch->pq[batch->pq_amount]) >= static_cast<size_t>(batch->pq_th))
            batch->pq_amount++;

        pthread_mutex_unlock(&batch->pq_insert_lock);
    }
    else
    {
        generate_pqs_of_rs_batch(subtree_node->right_child, batch, bsf_distance, paa, index);
        generate_pqs_of_rs_batch(subtree_node->left_child, batch, bsf_distance, paa, index);
    }
}

void process_rs_batch(int batch_index, SubtreeBatch *batches, float bsf_distance, isax_index *index, ts_type *paa)
{
    while (1)
    {
        int subtree_index = static_cast<int>(__atomic_fetch_add(reinterpret_cast<volatile int *>(&batches[batch_index].current_subtree_to_process), 1, __ATOMIC_SEQ_CST));

        if (subtree_index >= batches[batch_index].size)
            break;

        isax_node *subtree_node = (batches[batch_index].nodelist->nlist)[static_cast<size_t>(subtree_index + batches[batch_index].from)];
        generate_pqs_of_rs_batch(subtree_node, &batches[batch_index], bsf_distance, paa, index);
    }
}

void gather_sort_pqueues(QaWorkerData *in_data)
{
    int total_pqs = 0;

    for (int i = 0; i < in_data->total_batches; i++)
    {
        int pqs = in_data->batches[i].pq_amount + 1;
        for (int j = 0; j < pqs; j++)
        {
            if (in_data->batches[i].pq[j] != nullptr && pqueue_size(in_data->batches[i].pq[j]) > 0)
                total_pqs++;
        }
    }

    *(in_data->final_pq_list_size) = total_pqs;
    if (total_pqs == 0)
    {
        *(in_data->final_pq_list) = nullptr;
        *(in_data->priority_queues_filled) = 1;
        return;
    }

    *(in_data->final_pq_list) = static_cast<pqueue_t **>(std::malloc(sizeof(pqueue_t *) * static_cast<size_t>(total_pqs)));
    if (*(in_data->final_pq_list) == nullptr)
    {
        printf("MEMORY ERROR: final_pq_list allocation failed. Exiting...\n");
        std::exit(EXIT_FAILURE);
    }

    int curr_pq = 0;
    for (int i = 0; i < in_data->total_batches; i++)
    {
        int pqs = in_data->batches[i].pq_amount + 1;
        for (int j = 0; j < pqs; j++)
        {
            if (in_data->batches[i].pq[j] != nullptr)
            {
                if (pqueue_size(in_data->batches[i].pq[j]) > 0)
                {
                    in_data->batches[i].pq[j]->batch_id = i;
                    (*(in_data->final_pq_list))[curr_pq++] = in_data->batches[i].pq[j];
                }
                else
                {
                    pqueue_free(in_data->batches[i].pq[j]);
                    in_data->batches[i].pq[j] = nullptr;
                }
            }
        }
    }

    std::qsort(*(in_data->final_pq_list), static_cast<size_t>(total_pqs), sizeof(pqueue_t *), pq_comparator);

    for (int pq_num = 0; pq_num < total_pqs; pq_num++)
    {
        pqueue_t *p = (*(in_data->final_pq_list))[pq_num];
        SubtreeBatch *belonging_batch = &in_data->batches[p->batch_id];

        if (belonging_batch->min_pq_index > pq_num)
            belonging_batch->min_pq_index = pq_num;
        if (belonging_batch->max_pq_index < pq_num)
            belonging_batch->max_pq_index = pq_num;
    }

    *(in_data->priority_queues_filled) = 1;
}

// kNN: offer a verified window. With a merge offset it is recorded as a
// candidate when within the pruning bound (see g_knn_candidates).
static inline void offer_knn_window(QaWorkerData *in_data, pqueue_bsf *pq_bsf, float dist,
                                    file_position_type position, isax_node *node)
{
    pthread_mutex_lock(in_data->bsf_lock);
    const float bound = pq_bsf->knn[pq_bsf->k - 1];
    if (in_data->knn_candidates != nullptr && dist <= bound)
        in_data->knn_candidates->emplace_back(position, dist);
    if (dist < bound) // offset >= 1: a window verified twice (seed, then search) is kept once
        pqueue_bsf_insert_offset(pq_bsf, dist, position, node, std::max(1, in_data->merge_offset));
    pthread_mutex_unlock(in_data->bsf_lock);
}

int process_pq_of_batch_chatzakis(int current_pq_index, QaWorkerData *input_data)
{
    pqueue_t **final_pq_list = *(input_data->final_pq_list);
    int final_pq_list_size = *(input_data->final_pq_list_size);
    if (current_pq_index >= final_pq_list_size)
        return 0;
    pqueue_t *pq = final_pq_list[current_pq_index];
    if (pq == nullptr || pq->is_stolen)
        return 0;

    pqueue_bsf *pq_bsf = input_data->bsf_result->pq_bsf;
    query_result *n = static_cast<query_result *>(pqueue_pop(pq));

    if (n == nullptr)
        return 0;

    float bsf = pq_bsf->knn[pq_bsf->k - 1];
    if (n->distance > bsf || n->distance > input_data->minimum_distance)
    {
        std::free(n);
        return 0;
    }

    if (n->node->is_leaf)
    {
        isax_index *index = input_data->index;
        ts_type *query = input_data->ts;
        ts_type *paa = input_data->paa;
        float *rawfile = input_data->rawfile;
        isax_node *node = n->node;
        int my_rank = input_data->my_rank;
        const int ts_size = index->settings->timeseries_size;

        if (node->buffer != nullptr)
        {
            const int record_count = node->buffer->partial_buffer_size;
            std::vector<std::pair<file_position_type, float>> leaf_hits; // appended once per leaf, not per hit
            // Raw blocks prefetched per record follow the recent number of
            // blocks the early-abandoning distance needed.
            const int block_count = ts_size / ZNORM_BLOCK;
            int avg_blocks_x16 = 2 * 16;
            int blocks_visited = 0;
            file_position_type **position_buffer = node->buffer->partial_position_buffer;
            sax_type **sax_buffer = node->buffer->partial_sax_buffer;

            // Pass 1: lower bounds from the SAX words only (prefetched ahead),
            // skipped when the sampled bounds show that they cannot prune.
            static thread_local std::vector<int> passing;
            passing.clear();
            const bool record_lb = input_data->record_lb;
            for (int i = 0; i < record_count; i++)
            {
                if (record_lb && i + RECORD_PREFETCH_FAR < record_count && position_buffer[i + RECORD_PREFETCH_FAR] != nullptr)
                    __builtin_prefetch(sax_buffer[i + RECORD_PREFETCH_FAR]);
                if (position_buffer[i] != nullptr && (!record_lb || minidist_paa_to_raw_sax_safe(paa, sax_buffer[i], index) <= bsf))
                    passing.push_back(i);
            }

            // Pass 2: distances of the records that pass. They point to
            // scattered positions, so the loop is bound by memory latency:
            // prefetch the position of a record far ahead, then the statistics
            // and first visited raw blocks of a record close ahead.
            const int pass_count = static_cast<int>(passing.size());
            for (int j = 0; j < pass_count; j++)
            {
                if (j + RECORD_PREFETCH_FAR < pass_count)
                    __builtin_prefetch(position_buffer[passing[j + RECORD_PREFETCH_FAR]]);
                if (j + RECORD_PREFETCH_NEAR < pass_count)
                {
                    const file_position_type ahead = *position_buffer[passing[j + RECORD_PREFETCH_NEAR]];
                    if (ahead >= input_data->local_owned_subseq_start &&
                        ahead < input_data->local_owned_subseq_start + input_data->local_owned_subseq_count)
                    {
                        const idx_t ahead_id = static_cast<idx_t>(ahead - input_data->local_owned_subseq_start);
                        const float *ahead_raw = rawfile + static_cast<idx_t>(ahead - input_data->local_chunk_start);
                        __builtin_prefetch(&input_data->subsequence_means[ahead_id]);
                        __builtin_prefetch(&input_data->subsequence_stds[ahead_id]);
                        const int prefetch_blocks = std::min(std::min(block_count, RECORD_PREFETCH_MAX_BLOCKS),
                                                             (avg_blocks_x16 >> 4) + 1);
                        for (int b = 0; b < prefetch_blocks; b++)
                        {
                            __builtin_prefetch(ahead_raw + input_data->block_order[b]);
                            __builtin_prefetch(ahead_raw + input_data->block_order[b] + 16);
                        }
                    }
                }

                const file_position_type global_pos = *position_buffer[passing[j]];

                if (global_pos < input_data->local_owned_subseq_start ||
                    global_pos >= input_data->local_owned_subseq_start + input_data->local_owned_subseq_count)
                {
                    continue;
                }

                const idx_t local_subseq_id = static_cast<idx_t>(global_pos - input_data->local_owned_subseq_start);
                const idx_t local_raw_start = static_cast<idx_t>(global_pos - input_data->local_chunk_start);
                if (local_raw_start + static_cast<idx_t>(ts_size) > input_data->local_chunk_length)
                {
                    continue;
                }

                if (rawfile[local_raw_start] == TREMOR_MISSING_VALUE)
                {
                    continue;
                }

                const float mean = input_data->subsequence_means[local_subseq_id];
                const float stddev = input_data->subsequence_stds[local_subseq_id];

                float dist = l2_dist_znorm_subsequence(
                    query,
                    rawfile,
                    local_raw_start,
                    ts_size,
                    mean,
                    stddev,
                    input_data->threshold_mode ? input_data->threshold_ed : pq_bsf->knn[pq_bsf->k - 1],
                    input_data->block_order,
                    &blocks_visited);
                // Moving average (weight 1/16) in units of 1/16 block.
                avg_blocks_x16 += blocks_visited - (avg_blocks_x16 >> 4);

                if (input_data->threshold_mode)
                {
                    if (dist <= input_data->threshold_ed &&
                        input_data->threshold_hits != nullptr &&
                        input_data->threshold_hits_lock != nullptr)
                    {
                        leaf_hits.emplace_back(global_pos, dist);
                    }
                }
                else if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    offer_knn_window(input_data, pq_bsf, dist, global_pos, node);

                    bsf_sharing_recv_bsf(*input_data->bsf_sharing_data, pq_bsf, input_data->workernumber, *input_data->shared_bsf_results, input_data->bsf_lock, my_rank, input_data->comm_sz, input_data->query_counter);
                    bsf_sharing_bcast_bsf(*input_data->bsf_sharing_data, pq_bsf, input_data->workernumber, my_rank, input_data->query_counter, input_data->replication_data, nullptr);
                }
            }
            if (!leaf_hits.empty() && input_data->threshold_hits != nullptr && input_data->threshold_hits_lock != nullptr)
            {
                pthread_mutex_lock(input_data->threshold_hits_lock);
                input_data->threshold_hits->insert(input_data->threshold_hits->end(), leaf_hits.begin(), leaf_hits.end());
                pthread_mutex_unlock(input_data->threshold_hits_lock);
            }
        }
    }

    std::free(n);
    return 1;
}

void *workstealing_manager(void *rfdata)
{
    WorkstealingThreadData *data = static_cast<WorkstealingThreadData *>(rfdata);
    volatile char *threads_finished = data->query_workers_finished;
    static bool first_time = true;

    ReplicationData *replication_data = data->replication_data;
    WorkstealingData *workstealing_data = data->workstealing_data;
    int my_rank = data->my_rank;
    int query_counter = data->query_counter;

    int recv_message = 0;
    int ready = 0;

    pqueue_t **final_pq_list = *(data->final_pq_list);
    int final_pq_list_size = *(data->final_pq_list_size);
    int local_pqs_stolen = 0;
    int local_ws_times = 0;

    int coordinator_of_current_group_rank = rep_find_coordinator_node_rank(*replication_data, my_rank);
    int repgroup_nodes = rep_get_repgroup_nodes(*replication_data, my_rank);

    if (first_time)
    {
        for (int rank = coordinator_of_current_group_rank; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            if (rank == my_rank)
                continue;
            if (rank >= 0 && rank < static_cast<int>(workstealing_data->global_helper_requests.size()))
            {
                MPI_Irecv(&recv_message, 1, MPI_INT, rank, WORKSTEALING_INFORM_AVAILABILITY, MPI_COMM_WORLD, &workstealing_data->global_helper_requests[static_cast<size_t>(rank)]);
            }
        }
        first_time = false;
    }

    while (*threads_finished == 0)
    {
        for (int rank = coordinator_of_current_group_rank; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            if (rank == my_rank)
                continue;

            if (rank < 0 || rank >= static_cast<int>(workstealing_data->global_helper_requests.size()))
                continue;

            MPI_Test(&workstealing_data->global_helper_requests[static_cast<size_t>(rank)], &ready, MPI_STATUS_IGNORE);

            if (ready)
            {
                if (ENABLE_PRINTS_WORKSTEALING)
                {
                    printf("[WORKSTEALING MAIN - Node %d] - Workstealing thread found that node %d can help!\n", my_rank, rank);
                }

                *(data->receiving_workstealing) = 1;

                while (!(*data->priority_queues_filled))
                {
                }

                if (workstealing_data->ws_type == WorkstealingType::S_WS)
                {
                    std::vector<int> batch_ids_to_send(static_cast<size_t>(workstealing_data->items_to_send));
                    bool are_batches_available = true;

                    for (int i = 0; i < workstealing_data->items_to_send; i++)
                    {
                        int selected_batch_id = -1;
                        int farest_min = 0;
                        for (int b = 0; b < data->batchlist->batch_amount; b++)
                        {
                            if (data->batchlist->batches[b].is_stolen || data->batchlist->batches[b].pq[0] == nullptr)
                                continue;

                            int current_min = data->batchlist->batches[b].min_pq_index;
                            if (current_min >= farest_min)
                            {
                                farest_min = current_min;
                                selected_batch_id = b;
                            }
                        }

                        if (selected_batch_id == -1)
                        {
                            are_batches_available = false;
                            break;
                        }
                        batch_ids_to_send[static_cast<size_t>(i)] = selected_batch_id;
                        data->batchlist->batches[selected_batch_id].is_stolen = 1;
                    }

                    if (are_batches_available)
                    {
                        for (int id = 0; id < workstealing_data->items_to_send; id++)
                        {
                            int bid = batch_ids_to_send[static_cast<size_t>(id)];
                            for (int i = 0; i < data->batchlist->batches[bid].pq_amount + 1; i++)
                            {
                                if (data->batchlist->batches[bid].pq[i] != nullptr && data->batchlist->batches[bid].pq[i]->size > 1)
                                {
                                    data->batchlist->batches[bid].pq[i]->is_stolen = 1;
                                    local_pqs_stolen++;
                                    data->batchlist->batches[bid].pq[i]->is_processed = 1;
                                }
                            }
                        }

                        local_ws_times++;

                        float bsf_val = 0.0f;
                        file_position_type pos_val = 0;
                        if (data->bsf_result->pq_bsf != nullptr)
                        {
                            bsf_val = data->bsf_result->pq_bsf->knn[data->bsf_result->pq_bsf->k - 1];
                            pos_val = data->bsf_result->pq_bsf->position[data->bsf_result->pq_bsf->k - 1];
                        }

                        std::vector<unsigned long long> datas(static_cast<size_t>(3 + workstealing_data->items_to_send));
                        datas[0] = static_cast<unsigned long long>(query_counter);
                        datas[1] = pack_float_for_mpi(bsf_val);
                        datas[2] = static_cast<unsigned long long>(pos_val);
                        for (int i = 0; i < workstealing_data->items_to_send; i++)
                            datas[static_cast<size_t>(3 + i)] = static_cast<unsigned long long>(batch_ids_to_send[static_cast<size_t>(i)]);

                        if (ENABLE_PRINTS_WORKSTEALING)
                        {
                            printf("[WORKSTEALING MAIN NODE - Node %d]: Sending message to node %d :  [ %d, %f, %llu, ", my_rank, rank, (int)datas[0], bsf_val, (unsigned long long)datas[2]);
                            for (int t = 0; t < workstealing_data->items_to_send; t++)
                                printf("%d ", (int)datas[static_cast<size_t>(t + 3)]);
                            printf("]\n");
                        }

                        MPI_Send(datas.data(), 3 + workstealing_data->items_to_send, MPI_UNSIGNED_LONG_LONG, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD);
                    }
                    else
                    {
                        std::vector<unsigned long long> datas(static_cast<size_t>(3 + workstealing_data->items_to_send));
                        datas[0] = static_cast<unsigned long long>(-1);
                        datas[1] = 0;
                        datas[2] = 0;
                        MPI_Send(datas.data(), 3 + workstealing_data->items_to_send, MPI_UNSIGNED_LONG_LONG, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD);
                    }
                }
                else if (workstealing_data->ws_type == WorkstealingType::P_WS)
                {
                    final_pq_list_size = *(data->final_pq_list_size);
                    final_pq_list = *(data->final_pq_list);

                    std::vector<pqueue_t *> pq_candidates(static_cast<size_t>(workstealing_data->items_to_send));
                    int pq_candidates_size = 0;

                    if (ENABLE_PRINTS_WORKSTEALING)
                    {
                        printf("[WORKSTEALING MAIN NODE %d]: Total priority queue size: %d\n", my_rank, final_pq_list_size);
                    }

                    int segments = data->index->settings->paa_segments;
                    for (int pqindex = final_pq_list_size - local_pqs_stolen - 1; pqindex >= 0; pqindex--)
                    {
                        if (final_pq_list[pqindex]->is_stolen || final_pq_list[pqindex]->is_processed)
                            continue;

                        if (pq_candidates_size == workstealing_data->items_to_send)
                            break;

                        local_pqs_stolen++;
                        local_ws_times++;

                        final_pq_list[pqindex]->is_stolen = 1;
                        final_pq_list[pqindex]->is_processed = 1;

                        pq_candidates[static_cast<size_t>(pq_candidates_size)] = final_pq_list[pqindex];
                        pq_candidates_size++;
                    }

                    if (pq_candidates_size == 0)
                    {
                        int data_size = 2 + pq_candidates_size * (segments * 2);
                        std::vector<float> send_datas(static_cast<size_t>(data_size > 0 ? data_size : 1));
                        send_datas[0] = -1.0f;

                        if (ENABLE_PRINTS_WORKSTEALING)
                        {
                            printf("[WORKSTEALING MAIN NODE %d]: Nothing to send to node %d\n", my_rank, rank);
                        }
                        MPI_Send(send_datas.data(), data_size, MPI_FLOAT, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD);
                    }
                    else
                    {
                        int data_size = 2 + 1 + workstealing_data->items_to_send * (segments * 2);
                        std::vector<float> send_datas(static_cast<size_t>(data_size));

                        send_datas[0] = static_cast<float>(query_counter);
                        float bsf_for_pws = 0.0f;
                        if (data->bsf_result->pq_bsf != nullptr)
                            bsf_for_pws = data->bsf_result->pq_bsf->knn[data->bsf_result->pq_bsf->k - 1];
                        send_datas[1] = bsf_for_pws;
                        send_datas[2] = static_cast<float>(pq_candidates_size);

                        isax_index *data_index = data->index;

                        for (int pqindex = 0; pqindex < pq_candidates_size; pqindex++)
                        {
                            pqueue_t *pq = pq_candidates[static_cast<size_t>(pqindex)];

                            isax_node *starting_node = pq->starting_node;
                            isax_node *ending_node = pq->ending_node;
                            if (starting_node == nullptr || ending_node == nullptr)
                            {
                                printf("ERROR: Starting or ending node is NULL\n");
                                std::exit(EXIT_FAILURE);
                            }

                            isax_node *lca_node = ws_compute_lca(data_index, starting_node, ending_node);
                            if (lca_node == nullptr)
                            {
                                printf("ERROR: LCA node is NULL\n");
                                std::exit(EXIT_FAILURE);
                            }

                            for (int seg = 0; seg < segments; seg++)
                            {
                                send_datas[static_cast<size_t>(3 + (pqindex * (segments * 2)) + seg)] = static_cast<float>(lca_node->isax_values[seg]);
                                send_datas[static_cast<size_t>(3 + (pqindex * (segments * 2)) + (segments + seg))] = static_cast<float>(lca_node->isax_cardinalities[seg]);
                            }
                        }

                        if (ENABLE_PRINTS_WORKSTEALING)
                        {
                            printf("[WORKSTEALING MAIN NODE %d]: Sendingg %d priority queues to node %d\n", my_rank, pq_candidates_size, rank);
                        }

                        MPI_Send(send_datas.data(), data_size, MPI_FLOAT, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD);
                    }
                }

                MPI_Irecv(&recv_message, 1, MPI_INT, rank, WORKSTEALING_INFORM_AVAILABILITY, MPI_COMM_WORLD, &workstealing_data->global_helper_requests[static_cast<size_t>(rank)]);
            }
        }
    }

    *(data->pqs_stolen) += local_pqs_stolen;
    *(data->workstealing_times) += local_ws_times;

    pthread_exit(nullptr);
    return nullptr;
}

void *dynamic_query_scheduler(void *rfdata)
{
    CoordinatorData *data = static_cast<CoordinatorData *>(rfdata);

    CommunicationModuleData *comm_data = data->comm_data;
    volatile char *threads_finished = data->threads_finished;

    while (!(*threads_finished))
    {
        if (comm_data != nullptr)
        {
            call_module(comm_data);
        }
    }

    pthread_exit(nullptr);
    return nullptr;
}

// Attributes of query worker i: pinned to the i-th CPU this process may
// use. Unpinned, the scheduler first stacks the workers on few cores and
// the scan phase runs about 1.5x slower.
static pthread_attr_t *worker_attr(int i)
{
    static std::vector<int> cpus;
    static std::vector<pthread_attr_t> attrs;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    if (cpus.empty())
    {
        cpu_set_t set;
        sched_getaffinity(0, sizeof(set), &set);
        for (int c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, &set))
                cpus.push_back(c);
        attrs.resize(cpus.size());
        for (size_t j = 0; j < cpus.size(); j++)
        {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpus[j], &one);
            pthread_attr_init(&attrs[j]);
            pthread_attr_setaffinity_np(&attrs[j], sizeof(one), &one);
        }
    }
    return &attrs[static_cast<size_t>(i) % attrs.size()];
}

// iSAX lower bound between a query PAA and one window's full-cardinality word.
static inline float window_lower_bound(const isax_index_settings *settings, ts_type *paa, const sax_type *word)
{
    sax_type *w = const_cast<sax_type *>(word);
    if (settings->paa_segments == 16)
        return minidist_paa_to_isax_rawa_SIMD(paa, w, settings->max_sax_cardinalities, settings->sax_bit_cardinality,
                                              settings->sax_alphabet_cardinality, settings->paa_segments, MINVAL, MAXVAL,
                                              settings->mindist_sqrt);
    return minidist_paa_to_isax(paa, w, settings->max_sax_cardinalities, settings->sax_bit_cardinality,
                                settings->sax_alphabet_cardinality, settings->paa_segments, MINVAL, MAXVAL,
                                settings->mindist_sqrt);
}

// Persistent query workers (pinned with worker_attr) that run one query's
// qa_exact_search_tremor_worker calls: creating 128 threads per query
// costs several milliseconds, as much as a selective query itself.
class WorkerPool
{
public:
    // Runs fn(&items[i]) for i < n on workers 0..n-1; wait() joins them.
    void start(int n, void *(*fn)(void *), QaWorkerData *items)
    {
        std::lock_guard<std::mutex> guard(lock);
        while (static_cast<int>(size) < n)
        {
            pthread_t thread;
            if (pthread_create(&thread, worker_attr(size), &WorkerPool::entry, new Start{this, size}) != 0)
            {
                printf("Error creating query worker %d\n", size);
                std::exit(EXIT_FAILURE);
            }
            pthread_detach(thread);
            size++;
        }
        task = fn;
        task_items = items;
        active = n;
        pending = n;
        generation++;
        wake.notify_all();
    }

    void wait()
    {
        std::unique_lock<std::mutex> guard(lock);
        done.wait(guard, [this]
                  { return pending == 0; });
    }

private:
    struct Start
    {
        WorkerPool *pool;
        int id;
    };

    static void *entry(void *arg)
    {
        Start start = *static_cast<Start *>(arg);
        delete static_cast<Start *>(arg);
        start.pool->loop(start.id);
        return nullptr;
    }

    void loop(int id)
    {
        long long seen = 0;
        for (;;)
        {
            void *(*fn)(void *);
            QaWorkerData *item;
            {
                std::unique_lock<std::mutex> guard(lock);
                wake.wait(guard, [&]
                          { return generation != seen; });
                seen = generation;
                if (id >= active)
                    continue;
                fn = task;
                item = &task_items[id];
            }
            fn(item);
            std::lock_guard<std::mutex> guard(lock);
            if (--pending == 0)
                done.notify_all();
        }
    }

    std::mutex lock;
    std::condition_variable wake, done;
    int size = 0, active = 0, pending = 0;
    long long generation = 0;
    void *(*task)(void *) = nullptr;
    QaWorkerData *task_items = nullptr;
};

// Never destroyed: its idle workers wait on it until the process exits.
static WorkerPool &query_workers()
{
    static WorkerPool *pool = new WorkerPool();
    return *pool;
}

// Sampled candidate fraction from which the scan skips the lower bounds of
// single windows: reading a window's iSAX word and computing its bound costs
// about as much as abandoning its distance early, so the bounds pay off only
// when they prune almost every window (tuned on Maule and SeiFR: 0.05;
// TREMOR_SCAN_LB_FRACTION overrides it for experiments). In the leaves, a
// skipped bound means a distance at a scattered position, so they keep their
// bounds unless most windows pass (LEAF_LB_FRACTION).
static double scan_lb_fraction()
{
    static const double value = std::getenv("TREMOR_SCAN_LB_FRACTION") ? std::atof(std::getenv("TREMOR_SCAN_LB_FRACTION")) : 0.05;
    return value;
}
constexpr double LEAF_LB_FRACTION = 0.5;

// Adaptive fallback, a skip-sequential scan: visit every indexed window in
// position order (same eligibility as index construction), skip it when its
// iSAX lower bound exceeds the current bound (if use_lb), and otherwise
// compute its distance. Workers claim contiguous ranges, so consecutive
// windows share most of their samples in cache. Results equal
// the leaf-order search: threshold hits are sorted afterwards, and kNN
// answers do not depend on the visiting order.
static void refine_in_position_order(QaWorkerData *in_data, bool use_lb)
{
    constexpr long long RANGE = 1LL << 16; // windows per claim, a multiple of 64
    pqueue_bsf *pq_bsf = in_data->bsf_result->pq_bsf;
    const isax_index_settings *settings = in_data->index->settings;
    const int ts_size = settings->timeseries_size;
    const int paa_segments = settings->paa_segments;
    const long long owned_count = static_cast<long long>(in_data->local_owned_subseq_count);
    const idx_t owned_start = in_data->local_owned_subseq_start;
    const idx_t chunk_start = in_data->local_chunk_start;
    const idx_t chunk_length = in_data->local_chunk_length;
    const float *rawfile = in_data->rawfile;
    const float *means = in_data->subsequence_means;
    const float *stds = in_data->subsequence_stds;
    const sax_type *sax = g_subsequence_sax.data();
    ts_type *paa = in_data->paa;
    const ts_type *query = in_data->ts;
    const int *block_order = in_data->block_order;
    const bool threshold_mode = in_data->threshold_mode;
    const float threshold_ed = in_data->threshold_ed;
    CommunicationModuleData *comm_data = in_data->comm_data;
    std::vector<std::pair<file_position_type, float>> hits;
    int blocks_visited = 0;

    for (;;)
    {
        const long long begin = __atomic_fetch_add(in_data->refine_cursor, RANGE, __ATOMIC_RELAXED);
        if (begin >= owned_count)
            break;
        const long long end = std::min(begin + RANGE, owned_count);

        if (comm_data != nullptr && in_data->workernumber == 0 && comm_data->mode == DynamicSchedulingMode::PERIODIC_CHECK)
            call_module(comm_data);
        if (!threshold_mode)
            bsf_sharing_recv_bsf(*in_data->bsf_sharing_data, pq_bsf, in_data->workernumber, *in_data->shared_bsf_results,
                                 in_data->bsf_lock, in_data->my_rank, in_data->comm_sz, in_data->query_counter);

        for (long long id = begin; id < end; id++)
        {
            const file_position_type global_pos = static_cast<file_position_type>(owned_start + id);
            const idx_t local_raw_start = static_cast<idx_t>(global_pos - chunk_start);
            if (local_raw_start + static_cast<idx_t>(ts_size) > chunk_length)
                continue;
            if (rawfile[local_raw_start] == TREMOR_MISSING_VALUE)
                continue;

            const float bound = threshold_mode ? threshold_ed : pq_bsf->knn[pq_bsf->k - 1];
            if (use_lb && window_lower_bound(settings, paa, &sax[static_cast<size_t>(id) * paa_segments]) > bound)
                continue;

            const float dist = l2_dist_znorm_subsequence(query, rawfile, local_raw_start, ts_size, means[id], stds[id],
                                                         bound, block_order, &blocks_visited);

            if (threshold_mode)
            {
                if (dist <= threshold_ed)
                    hits.emplace_back(global_pos, dist);
            }
            else if (dist <= pq_bsf->knn[pq_bsf->k - 1])
            {
                offer_knn_window(in_data, pq_bsf, dist, global_pos, nullptr);

                bsf_sharing_recv_bsf(*in_data->bsf_sharing_data, pq_bsf, in_data->workernumber, *in_data->shared_bsf_results,
                                     in_data->bsf_lock, in_data->my_rank, in_data->comm_sz, in_data->query_counter);
                bsf_sharing_bcast_bsf(*in_data->bsf_sharing_data, pq_bsf, in_data->workernumber, in_data->my_rank,
                                      in_data->query_counter, in_data->replication_data, nullptr);
            }
        }
    }

    if (!hits.empty() && in_data->threshold_hits != nullptr && in_data->threshold_hits_lock != nullptr)
    {
        pthread_mutex_lock(in_data->threshold_hits_lock);
        in_data->threshold_hits->insert(in_data->threshold_hits->end(), hits.begin(), hits.end());
        pthread_mutex_unlock(in_data->threshold_hits_lock);
    }
}

// RefinementMode::Adaptive decision: count the windows of a strided sample
// of this rank's windows (this worker's share) whose own iSAX lower bound
// is within `bound`, i.e., the windows whose distance both strategies
// compute. The leaf search reaches them at scattered positions, the scan
// in waveform order, so the scan wins when their share exceeds the scan
// fallback fraction.
constexpr long long ADAPTIVE_SAMPLE = 1LL << 16;

static long long adaptive_sample_size(const QaWorkerData *in_data)
{
    return std::min<long long>(ADAPTIVE_SAMPLE, static_cast<long long>(in_data->local_owned_subseq_count));
}

static long long sample_candidates(const QaWorkerData *in_data, float bound)
{
    const long long owned = static_cast<long long>(in_data->local_owned_subseq_count);
    const long long samples = adaptive_sample_size(in_data);
    const long long stride = samples > 0 ? owned / samples : 1;
    long long candidates = 0;
    for (long long j = in_data->workernumber; j < samples; j += in_data->query_threads)
    {
        const idx_t id = static_cast<idx_t>(j * stride);
        if (window_lower_bound(in_data->index->settings, in_data->paa,
                               &g_subsequence_sax[id * static_cast<idx_t>(in_data->index->settings->paa_segments)]) <= bound)
            candidates++;
    }
    return candidates;
}

// Scan cost estimate: blocks that the early-abandoning kernel visits on this worker's share of a strided sample
// of the windows, with the current bound (nothing is recorded).
constexpr long long BLOCK_SAMPLE = 1LL << 13;

static long long sample_blocks(const QaWorkerData *in_data, float bound)
{
    const long long owned = static_cast<long long>(in_data->local_owned_subseq_count);
    const long long samples = std::min(BLOCK_SAMPLE, owned);
    const long long stride = samples > 0 ? owned / samples : 1;
    const int ts_size = in_data->index->settings->timeseries_size;
    long long blocks = 0;
    for (long long j = in_data->workernumber; j < samples; j += in_data->query_threads)
    {
        const idx_t id = static_cast<idx_t>(j * stride);
        const idx_t raw_start = static_cast<idx_t>(in_data->local_owned_subseq_start + id - in_data->local_chunk_start);
        if (raw_start + static_cast<idx_t>(ts_size) > in_data->local_chunk_length || in_data->rawfile[raw_start] == TREMOR_MISSING_VALUE)
            continue;
        int visited = 0;
        l2_dist_znorm_subsequence(in_data->ts, in_data->rawfile, raw_start, ts_size, in_data->subsequence_means[id],
                                  in_data->subsequence_stds[id], bound, in_data->block_order, &visited);
        blocks += visited;
    }
    return blocks;
}

// FFT fallback (fft_fallback.hpp): the FFT distances filter the windows, and the windows that pass are verified
// with the exact kernel and recorded as in refine_in_position_order, so the answers are those of the scan.
static void fft_in_position_order(QaWorkerData *in_data)
{
    pqueue_bsf *pq_bsf = in_data->bsf_result->pq_bsf;
    const int ts_size = in_data->index->settings->timeseries_size;
    const idx_t owned_start = in_data->local_owned_subseq_start;
    const float *rawfile = in_data->rawfile;
    const float *means = in_data->subsequence_means;
    const float *stds = in_data->subsequence_stds;
    const ts_type *query = in_data->ts;
    const int *block_order = in_data->block_order;
    const bool threshold_mode = in_data->threshold_mode;
    const float threshold_ed = in_data->threshold_ed;
    CommunicationModuleData *comm_data = in_data->comm_data;
    std::vector<std::pair<file_position_type, float>> hits;
    int blocks_visited = 0;

    tremor_fft::pass(
        in_data->refine_cursor,
        [&]
        {
            if (comm_data != nullptr && in_data->workernumber == 0 && comm_data->mode == DynamicSchedulingMode::PERIODIC_CHECK)
                call_module(comm_data);
            if (!threshold_mode)
                bsf_sharing_recv_bsf(*in_data->bsf_sharing_data, pq_bsf, in_data->workernumber, *in_data->shared_bsf_results,
                                     in_data->bsf_lock, in_data->my_rank, in_data->comm_sz, in_data->query_counter);
        },
        [&]
        { return threshold_mode ? threshold_ed : pq_bsf->knn[pq_bsf->k - 1]; },
        [&](idx_t id, idx_t raw_start)
        {
            const file_position_type global_pos = static_cast<file_position_type>(owned_start + id);
            const float bound = threshold_mode ? threshold_ed : pq_bsf->knn[pq_bsf->k - 1];
            const float dist = l2_dist_znorm_subsequence(query, rawfile, raw_start, ts_size, means[id], stds[id], bound,
                                                         block_order, &blocks_visited);
            if (threshold_mode)
            {
                if (dist <= threshold_ed)
                    hits.emplace_back(global_pos, dist);
            }
            else if (dist <= pq_bsf->knn[pq_bsf->k - 1])
            {
                offer_knn_window(in_data, pq_bsf, dist, global_pos, nullptr);
                bsf_sharing_recv_bsf(*in_data->bsf_sharing_data, pq_bsf, in_data->workernumber, *in_data->shared_bsf_results,
                                     in_data->bsf_lock, in_data->my_rank, in_data->comm_sz, in_data->query_counter);
                bsf_sharing_bcast_bsf(*in_data->bsf_sharing_data, pq_bsf, in_data->workernumber, in_data->my_rank,
                                      in_data->query_counter, in_data->replication_data, nullptr);
            }
        });

    if (!hits.empty() && in_data->threshold_hits != nullptr && in_data->threshold_hits_lock != nullptr)
    {
        pthread_mutex_lock(in_data->threshold_hits_lock);
        in_data->threshold_hits->insert(in_data->threshold_hits->end(), hits.begin(), hits.end());
        pthread_mutex_unlock(in_data->threshold_hits_lock);
    }
}

// kNN: this worker's share of the windows of the query's own leaf, verified
// before anything else to give a first bound (see query_leaf). They are
// recorded as candidates, since the exact search may skip a leaf whose
// lower bound equals the final bound.
static void seed_knn_bound(QaWorkerData *in_data)
{
    const isax_node *leaf = in_data->seed_leaf;
    pqueue_bsf *pq_bsf = in_data->bsf_result->pq_bsf;
    const int ts_size = in_data->index->settings->timeseries_size;
    int blocks_visited = 0;
    for (int i = in_data->workernumber; i < leaf->buffer->partial_buffer_size; i += in_data->query_threads)
    {
        if (leaf->buffer->partial_position_buffer[i] == nullptr)
            continue;
        const file_position_type pos = *leaf->buffer->partial_position_buffer[i];
        if (pos < in_data->local_owned_subseq_start || pos >= in_data->local_owned_subseq_start + in_data->local_owned_subseq_count)
            continue;
        const idx_t id = static_cast<idx_t>(pos - in_data->local_owned_subseq_start);
        const idx_t raw_start = static_cast<idx_t>(pos - in_data->local_chunk_start);
        if (raw_start + static_cast<idx_t>(ts_size) > in_data->local_chunk_length || in_data->rawfile[raw_start] == TREMOR_MISSING_VALUE)
            continue;
        const float dist = l2_dist_znorm_subsequence(in_data->ts, in_data->rawfile, raw_start, ts_size,
                                                     in_data->subsequence_means[id], in_data->subsequence_stds[id],
                                                     pq_bsf->knn[pq_bsf->k - 1], in_data->block_order, &blocks_visited);
        if (dist <= pq_bsf->knn[pq_bsf->k - 1])
            offer_knn_window(in_data, pq_bsf, dist, pos, nullptr);
    }
}

// kNN without replication: the template's close matches lie on one node,
// so every rank adds the first bound windows of all ranks to its own
// bound set. Any pairwise-separated set of windows bounds the answer, so
// every rank can prune with the best first bound found anywhere.
static void share_knn_bound(QaWorkerData *in_data)
{
    pqueue_bsf *pq_bsf = in_data->bsf_result->pq_bsf;
    const int n = pq_bsf->k;
    std::vector<float> distances(pq_bsf->knn, pq_bsf->knn + n);
    std::vector<long long> positions(pq_bsf->position, pq_bsf->position + n);
    std::vector<float> all_distances(static_cast<size_t>(n) * in_data->comm_sz);
    std::vector<long long> all_positions(static_cast<size_t>(n) * in_data->comm_sz);
    MPI_Allgather(distances.data(), n, MPI_FLOAT, all_distances.data(), n, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(positions.data(), n, MPI_LONG_LONG, all_positions.data(), n, MPI_LONG_LONG, MPI_COMM_WORLD);
    pthread_mutex_lock(in_data->bsf_lock);
    for (size_t i = 0; i < all_distances.size(); i++)
    {
        if (all_distances[i] < pq_bsf->knn[n - 1])
            pqueue_bsf_insert_offset(pq_bsf, all_distances[i], static_cast<file_position_type>(all_positions[i]), nullptr,
                                     in_data->merge_offset);
    }
    pthread_mutex_unlock(in_data->bsf_lock);
}

// RefinementMode::Adaptive: share of the sampled windows within `bound`;
// every worker takes part and gets the same value.
static double adaptive_sampled_fraction(QaWorkerData *in_data, float bound)
{
    __atomic_fetch_add(in_data->adaptive_sample_candidates, sample_candidates(in_data, bound), __ATOMIC_RELAXED);
    pthread_barrier_wait(in_data->sync_barrier);
    const double fraction = static_cast<double>(*in_data->adaptive_sample_candidates) /
                            static_cast<double>(std::max(1LL, adaptive_sample_size(in_data)));
    pthread_barrier_wait(in_data->sync_barrier);
    return fraction;
}

void *qa_exact_search_tremor_worker(void *rfdata)
{
    QaWorkerData *in_data = static_cast<QaWorkerData *>(rfdata);
    CommunicationModuleData *comm_data = in_data->comm_data;
    int my_rank = in_data->my_rank;
    int comm_sz = in_data->comm_sz;
    int query_counter = in_data->query_counter;
    BsfSharingData *bsf_sharing_data = in_data->bsf_sharing_data;

    int k = in_data->bsf_result->pq_bsf->k;

    if (in_data->seed_leaf != nullptr)
    {
        seed_knn_bound(in_data);
        pthread_barrier_wait(in_data->sync_barrier);
    }
    if (in_data->share_seed)
    {
        if (in_data->workernumber == 0)
            share_knn_bound(in_data);
        pthread_barrier_wait(in_data->sync_barrier);
    }

    // Share of sampled windows within the bound: the lower bounds of single
    // windows are worth computing only if they prune (scan_lb_fraction).
    double fraction = 0.0;
    bool bounds_prune = true;
    if (in_data->adaptive_sample_candidates != nullptr)
    {
        fraction = adaptive_sampled_fraction(in_data, in_data->bsf_result->pq_bsf->knn[k - 1]);
        bounds_prune = fraction < scan_lb_fraction();
        // A kNN bound tightens quickly during the leaf search, so the leaves keep them.
        in_data->record_lb = fraction < LEAF_LB_FRACTION || !in_data->threshold_mode;
    }
    if (in_data->adaptive_decision != nullptr)
    {
        // RefinementMode::Adaptive: scan if too many sampled windows pass the
        // lower bound (see sample_candidates), else search the index.
        const bool scan = fraction > g_scan_fallback_fraction;
        if (in_data->workernumber == 0)
        {
            *(in_data->adaptive_decision) = scan ? 1 : 0;
            (scan ? g_adaptive_scan_queries : g_adaptive_leaf_queries)++;
            std::printf("[Node %d] adaptive query %d: sampled candidate fraction %.6f -> %s\n",
                        in_data->my_rank, in_data->query_counter, fraction, scan ? "scan" : "leaf");
        }
        if (scan)
        {
            if (in_data->workernumber == 0)
                *(in_data->priority_queues_filled) = 1;
            // Scan or FFT (fft_fallback.hpp): whichever the calibrated costs predict to be cheaper, from the
            // blocks that the scan visits per window on a sample, with the current bound.
            if (tremor_fft::available() && in_data->adaptive_sample_blocks != nullptr)
            {
                const float bound = in_data->threshold_mode ? in_data->threshold_ed : in_data->bsf_result->pq_bsf->knn[k - 1];
                __atomic_fetch_add(in_data->adaptive_sample_blocks, sample_blocks(in_data, bound), __ATOMIC_RELAXED);
                pthread_barrier_wait(in_data->sync_barrier);
                if (in_data->workernumber == 0)
                {
                    const long long samples = std::min(BLOCK_SAMPLE, static_cast<long long>(in_data->local_owned_subseq_count));
                    const double blocks = static_cast<double>(*in_data->adaptive_sample_blocks) / static_cast<double>(std::max(1LL, samples));
                    const double scan_cost = tremor_fft::scan_seconds(blocks), fft_cost = tremor_fft::fft_seconds();
                    const bool fft = fft_cost < 0.9 * scan_cost;
                    if (fft)
                    {
                        *(in_data->adaptive_decision) = 2;
                        tremor_fft::prepare_query(in_data->ts);
                        *(in_data->refine_cursor) = 0;
                    }
                    std::printf("[Node %d] adaptive query %d: scan %.4f s or FFT %.4f s (%.1f blocks per window) -> %s\n",
                                in_data->my_rank, in_data->query_counter, scan_cost, fft_cost, blocks, fft ? "fft" : "scan");
                }
                pthread_barrier_wait(in_data->sync_barrier);
            }
            if (*(in_data->adaptive_decision) == 2)
                fft_in_position_order(in_data);
            else
                refine_in_position_order(in_data, bounds_prune);
            pthread_barrier_wait(in_data->sync_barrier);
            return nullptr;
        }
    }

    for (;;)
    {
        int current_batch_index = __sync_fetch_and_add(in_data->batch_counter, 1);
        if (current_batch_index >= in_data->total_batches)
            break;

        if (comm_data != nullptr && in_data->workernumber == 0 && comm_data->mode == DynamicSchedulingMode::PERIODIC_CHECK)
        {
            call_module(comm_data);
        }

        float bsf = in_data->bsf_result->pq_bsf->knn[k - 1];
        process_rs_batch(current_batch_index, in_data->batches, bsf, in_data->index, in_data->paa);
        in_data->batches[current_batch_index].processed_phase_1 = 1;
        bsf_sharing_recv_bsf(*bsf_sharing_data, in_data->bsf_result->pq_bsf, in_data->workernumber, *in_data->shared_bsf_results, in_data->bsf_lock, my_rank, comm_sz, query_counter);
    }

    for (int i = 0; i < in_data->total_batches; i++)
    {
        if (!in_data->batches[i].processed_phase_1 && !in_data->batches[i].is_getting_help_phase1)
        {
            in_data->batches[i].is_getting_help_phase1 = 1;
            int k_help = in_data->bsf_result->pq_bsf->k;
            process_rs_batch(i, in_data->batches, in_data->bsf_result->pq_bsf->knn[k_help - 1], in_data->index, in_data->paa);
            bsf_sharing_recv_bsf(*bsf_sharing_data, in_data->bsf_result->pq_bsf, in_data->workernumber, *in_data->shared_bsf_results, in_data->bsf_lock, my_rank, comm_sz, query_counter);
        }
    }

    pthread_barrier_wait(in_data->sync_barrier);

    if (in_data->workernumber == 0)
    {
        gather_sort_pqueues(in_data);
    }

    pthread_barrier_wait(in_data->sync_barrier);

    for (;;)
    {
        int current_pq_index = __sync_fetch_and_add(in_data->pq_counter, 1);
        if (current_pq_index >= *(in_data->final_pq_list_size))
            break;

        pqueue_t *pq = (*(in_data->final_pq_list))[current_pq_index];
        if (pq != nullptr && pq->is_stolen)
            continue;

        while (process_pq_of_batch_chatzakis(current_pq_index, in_data))
        {
            bsf_sharing_recv_bsf(*bsf_sharing_data, in_data->bsf_result->pq_bsf, in_data->workernumber, *in_data->shared_bsf_results, in_data->bsf_lock, my_rank, comm_sz, query_counter);
            bsf_sharing_bcast_bsf(*bsf_sharing_data, in_data->bsf_result->pq_bsf, in_data->workernumber, my_rank, query_counter, in_data->replication_data, nullptr);

            if (comm_data != nullptr && in_data->workernumber == 0 && comm_data->mode == DynamicSchedulingMode::PERIODIC_CHECK)
            {
                call_module(comm_data);
            }
        }

        if (pq != nullptr)
            pq->is_processed = 1;
    }


    pthread_barrier_wait(in_data->sync_barrier);
    return nullptr;
}

// kNN approximate answer (classic iSAX): the leaf that the query's own iSAX
// word falls into. The workers verify its windows first (seed_knn_bound).
static isax_node *query_leaf(isax_index *index, ts_type *paa)
{
    const int paa_segments = index->settings->paa_segments;
    std::vector<sax_type> sax(static_cast<size_t>(paa_segments));
    sax_from_paa(paa, sax.data(), paa_segments, index->settings->sax_alphabet_cardinality,
                 index->settings->sax_bit_cardinality);
    root_mask_type root_mask = 0;
    CREATE_MASK(root_mask, index, sax.data());
    auto *fbl = reinterpret_cast<parallel_first_buffer_layer_ekosmas *>(index->fbl);
    if (fbl == nullptr || !fbl->soft_buffers[static_cast<int>(root_mask)].initialized)
        return nullptr;

    isax_node *node = fbl->soft_buffers[static_cast<int>(root_mask)].node;
    while (node != nullptr && !node->is_leaf && node->split_data != nullptr)
    {
        const int location = index->settings->sax_bit_cardinality - 1 -
                             node->split_data->split_mask[node->split_data->splitpoint];
        node = (sax[static_cast<size_t>(node->split_data->splitpoint)] & index->settings->bit_masks[location])
                   ? node->right_child
                   : node->left_child;
    }
    return node != nullptr && node->is_leaf && node->buffer != nullptr ? node : nullptr;
}

query_result qa_exact_search_tremor_knn(SearchFunctionParams args)
{
    int query_id = args.query_id;
    ts_type *ts = args.ts;
    ts_type *paa = args.paa;
    isax_index *index = args.index;
    NodeList *nodelist = args.nodelist;
    float minimum_distance = args.minimum_distance;
    double (*estimation_func)(double) = args.estimation_func;
    CommunicationModuleData *comm_data = args.comm_data;
    int k = args.k;
    pqueue_bsf *precomputed_bsfs = args.precomputed_bsfs;

    float *rawfile = args.rawfile;
    int merge_offset = args.merge_offset;
    int my_rank = args.my_rank;
    int query_threads = args.query_threads;
    int pq_th_div_factor = args.pq_th_div_factor;
    int comm_sz = args.comm_sz;
    BsfSharingData *bsf_sharing_data = args.bsf_sharing_data;
    WorkstealingData *workstealing_data = args.workstealing_data;
    ReplicationData *replication_data = args.replication_data;

    args.query_counter = query_id;
    int query_counter = args.query_counter;
    const double query_start = MPI_Wtime();
    const bool threshold_mode = args.threshold_mode;

    FILE *threshold_output_file = nullptr;
    pthread_mutex_t threshold_output_lock;
    idx_t threshold_match_counter = 0;
    std::vector<std::pair<file_position_type, float>> threshold_hits;
    pthread_mutex_t threshold_hits_lock;
    bool threshold_output_lock_initialized = false;
    bool threshold_hits_lock_initialized = false;
    if (threshold_mode)
    {
        pthread_mutex_init(&threshold_hits_lock, nullptr);
        threshold_hits_lock_initialized = true;
    }
    if (threshold_mode && !args.output_file.empty())
    {
        if (g_threshold_output == nullptr)
        {
            char node_output_name[512];
            std::snprintf(node_output_name, sizeof(node_output_name), "%s_%d.csv", args.output_file.c_str(), my_rank);
            g_threshold_output = std::fopen(node_output_name, "a");
            if (g_threshold_output == nullptr)
            {
                std::fprintf(stderr, "[Node %d] Error: Cannot open threshold output file %s\n", my_rank, node_output_name);
                std::exit(EXIT_FAILURE);
            }
            std::fseek(g_threshold_output, 0, SEEK_END);
            if (std::ftell(g_threshold_output) == 0)
            {
                std::fprintf(g_threshold_output, "Query, Counter, Position, Cross Correlation\n");
            }
        }
        threshold_output_file = g_threshold_output;
        pthread_mutex_init(&threshold_output_lock, nullptr);
        threshold_output_lock_initialized = true;
    }

    std::vector<BsfMessage> *shared_bsf_results = args.shared_bsf_results;

    query_result bsf_result;
    bsf_result.distance = FLT_MAX;
    bsf_result.node = nullptr;
    bsf_result.pqueue_position = 0;
    bsf_result.pq_bsf = nullptr;
    bsf_result.total_time = 0.0f;

    if (!precomputed_bsfs)
    {
        bsf_result.pq_bsf = threshold_mode
                                ? pqueue_bsf_init_from_val(k, args.threshold_ed, static_cast<file_position_type>(-1))
                                : pqueue_bsf_init(k);
    }
    else
    {
        bsf_result.pq_bsf = pqueue_bsf_init_from_src(k, precomputed_bsfs);
    }
    // kNN with a merge offset: 2k-1 pairwise-separated windows bound the
    // answer (see g_knn_candidates), and every window within the bound is a
    // candidate. The windows of the query's own leaf give the first bound.
    const bool knn_exclusion = !threshold_mode && merge_offset > 0;
    const std::vector<int> block_order = query_block_order(ts, index->settings->timeseries_size);
    std::vector<KnnCandidate> knn_candidates;
    if (knn_exclusion)
    {
        pqueue_bsf_destroy(bsf_result.pq_bsf);
        bsf_result.pq_bsf = pqueue_bsf_init(2 * k - 1);
    }
    isax_node *seed_leaf = knn_exclusion ? query_leaf(index, paa) : nullptr;
    // Without replication every rank answers every query in the same order,
    // so the ranks can share their first bounds (share_knn_bound).
    const bool share_seed = knn_exclusion && comm_sz > 1 && rep_get_repgroup_nodes(*replication_data, my_rank) == 1;

    // NOTE: rerank_pq_exact_l2 is intentionally not invoked here.
    // It assumed `rawfile` was an array of independent dim-length vectors
    // (FAISS-style), but in Tremor's long-sequence pipeline `rawfile` is
    // the local chunk of one long time series and `pq_bsf->position[i]`
    // are global subsequence offsets, not vector indices. The exact L2
    // rerank against the raw subsequences is performed later by
    // process_pq_of_batch_chatzakis via l2_dist_znorm_subsequence using
    // local_chunk_start / subsequence_means / subsequence_stds.
    (void)rawfile;

    if (!threshold_mode && !knn_exclusion && bsf_result.pq_bsf->knn[k - 1] == 0.0f)
    {
        return bsf_result;
    }

    int median = estimate_th(bsf_result.pq_bsf->knn[bsf_result.pq_bsf->k - 1], estimation_func);
    int query_th = median / pq_th_div_factor;
    if (query_th <= 0)
        query_th = 1;

    BatchList *batchlist = create_subtree_batches(nodelist, query_threads, query_th);

    pthread_t workstealing_thread = 0;
    pthread_t coordinator_thread_id = 0;

    pthread_barrier_t sync_barrier;
    pthread_barrier_init(&sync_barrier, nullptr, query_threads);

    pthread_mutex_t bsf_lock;
    pthread_mutex_t distances_lock;
    pthread_mutex_init(&bsf_lock, nullptr);
    pthread_mutex_init(&distances_lock, nullptr);

    volatile int batch_counter = 0;
    volatile int pq_counter = 0;
    volatile char query_workers_finished = 0;
    volatile char receiving_workstealing = 0;
    volatile char priority_queues_filled = 0;

    SubtreeBatch *batches = batchlist->batches;
    int total_batches = batchlist->batch_amount;

    pqueue_t **all_pqs = nullptr;
    int all_pqs_size = 0;
    int stolen_pqs = 0;
    int processed_pqs = 0;

    bsf_sharing_update_from_bookkeeping(*bsf_sharing_data, bsf_result.pq_bsf, *shared_bsf_results, query_counter);

    const bool adaptive = g_refinement_mode == RefinementMode::Adaptive;
    volatile long long adaptive_sample_candidates = 0;
    volatile long long adaptive_sample_blocks = 0;
    volatile long long refine_cursor = 0;
    volatile int adaptive_decision = 0;
    std::vector<QaWorkerData> workerdata(static_cast<size_t>(query_threads));
    for (int i = 0; i < query_threads; i++)
    {
        workerdata[static_cast<size_t>(i)].workernumber = i;
        workerdata[static_cast<size_t>(i)].ts = ts;
        workerdata[static_cast<size_t>(i)].block_order = block_order.data();
        workerdata[static_cast<size_t>(i)].adaptive_sample_candidates = &adaptive_sample_candidates;
        workerdata[static_cast<size_t>(i)].adaptive_sample_blocks = &adaptive_sample_blocks;
        workerdata[static_cast<size_t>(i)].seed_leaf = seed_leaf;
        workerdata[static_cast<size_t>(i)].record_lb = true;
        workerdata[static_cast<size_t>(i)].share_seed = share_seed;
        workerdata[static_cast<size_t>(i)].refine_cursor = &refine_cursor;
        workerdata[static_cast<size_t>(i)].knn_candidates = knn_exclusion ? &knn_candidates : nullptr;
        workerdata[static_cast<size_t>(i)].adaptive_decision = adaptive ? &adaptive_decision : nullptr;
        workerdata[static_cast<size_t>(i)].paa = paa;
        workerdata[static_cast<size_t>(i)].minimum_distance = minimum_distance;
        workerdata[static_cast<size_t>(i)].bsf_result = &bsf_result;
        workerdata[static_cast<size_t>(i)].batches = batches;
        workerdata[static_cast<size_t>(i)].total_batches = total_batches;
        workerdata[static_cast<size_t>(i)].batch_counter = &batch_counter;
        workerdata[static_cast<size_t>(i)].sync_barrier = &sync_barrier;
        workerdata[static_cast<size_t>(i)].bsf_lock = &bsf_lock;
        workerdata[static_cast<size_t>(i)].receiving_workstealing = &receiving_workstealing;
        workerdata[static_cast<size_t>(i)].index = index;
        workerdata[static_cast<size_t>(i)].pq_counter = &pq_counter;
        workerdata[static_cast<size_t>(i)].final_pq_list = &all_pqs;
        workerdata[static_cast<size_t>(i)].final_pq_list_size = &all_pqs_size;
        workerdata[static_cast<size_t>(i)].distances_lock = &distances_lock;
        workerdata[static_cast<size_t>(i)].priority_queues_filled = &priority_queues_filled;
        workerdata[static_cast<size_t>(i)].pqs_stolen = &stolen_pqs;
        workerdata[static_cast<size_t>(i)].processed_pqs = &processed_pqs;
        workerdata[static_cast<size_t>(i)].comm_data = comm_data;
        workerdata[static_cast<size_t>(i)].shared_bsf_results = shared_bsf_results;
        workerdata[static_cast<size_t>(i)].query_counter = query_counter;
        workerdata[static_cast<size_t>(i)].replication_data = replication_data;
        workerdata[static_cast<size_t>(i)].workstealing_data = workstealing_data;
        workerdata[static_cast<size_t>(i)].query_threads = query_threads;
        workerdata[static_cast<size_t>(i)].comm_sz = comm_sz;
        workerdata[static_cast<size_t>(i)].my_rank = my_rank;
        workerdata[static_cast<size_t>(i)].merge_offset = merge_offset;
        workerdata[static_cast<size_t>(i)].bsf_sharing_data = bsf_sharing_data;
        workerdata[static_cast<size_t>(i)].rawfile = rawfile;
        workerdata[static_cast<size_t>(i)].pq_th_div_factor = pq_th_div_factor;
        workerdata[static_cast<size_t>(i)].corr_threshold = args.corr_threshold;
        workerdata[static_cast<size_t>(i)].verbose = args.verbose;
        workerdata[static_cast<size_t>(i)].output_file = args.output_file;
        workerdata[static_cast<size_t>(i)].warp_window = args.warp_window;
        workerdata[static_cast<size_t>(i)].paaU = args.paaU;
        workerdata[static_cast<size_t>(i)].paaL = args.paaL;
        workerdata[static_cast<size_t>(i)].local_chunk_start = args.local_chunk_start;
        workerdata[static_cast<size_t>(i)].local_chunk_length = args.local_chunk_length;
        workerdata[static_cast<size_t>(i)].local_owned_subseq_start = args.local_owned_subseq_start;
        workerdata[static_cast<size_t>(i)].local_owned_subseq_count = args.local_owned_subseq_count;
        workerdata[static_cast<size_t>(i)].subsequence_means = args.subsequence_means;
        workerdata[static_cast<size_t>(i)].subsequence_stds = args.subsequence_stds;
        workerdata[static_cast<size_t>(i)].subsequence_length = args.subsequence_length;
        workerdata[static_cast<size_t>(i)].threshold_mode = threshold_mode;
        workerdata[static_cast<size_t>(i)].threshold_ed = args.threshold_ed;
        workerdata[static_cast<size_t>(i)].threshold_output_file = threshold_output_file;
        workerdata[static_cast<size_t>(i)].threshold_output_lock = threshold_output_lock_initialized ? &threshold_output_lock : nullptr;
        workerdata[static_cast<size_t>(i)].threshold_match_counter = threshold_output_file ? &threshold_match_counter : nullptr;
        workerdata[static_cast<size_t>(i)].threshold_hits = threshold_mode ? &threshold_hits : nullptr;
        workerdata[static_cast<size_t>(i)].threshold_hits_lock = threshold_hits_lock_initialized ? &threshold_hits_lock : nullptr;
    }
    query_workers().start(query_threads, qa_exact_search_tremor_worker, workerdata.data());

    int ws_pqs_stolen = 0;
    int ws_times = 0;
    WorkstealingThreadData ws_th_data;
    if (!threshold_mode && comm_sz > 1 && static_cast<int>(workstealing_data->ws_type) > 0)
    {
        ws_th_data.query_workers_finished = &query_workers_finished;
        ws_th_data.receiving_workstealing = &receiving_workstealing;
        ws_th_data.priority_queues_filled = &priority_queues_filled;
        ws_th_data.bsf_lock = &bsf_lock;
        ws_th_data.bsf_result = &bsf_result;
        ws_th_data.batchlist = batchlist;
        ws_th_data.pqs_stolen = &ws_pqs_stolen;
        ws_th_data.workstealing_times = &ws_times;
        ws_th_data.final_pq_list = &all_pqs;
        ws_th_data.final_pq_list_size = &all_pqs_size;
        ws_th_data.index = index;
        ws_th_data.my_rank = my_rank;
        ws_th_data.comm_sz = comm_sz;
        ws_th_data.rawfile = rawfile;
        ws_th_data.query_counter = query_counter;
        ws_th_data.merge_offset = merge_offset;
        ws_th_data.replication_data = replication_data;
        ws_th_data.workstealing_data = workstealing_data;
        ws_th_data.bsf_sharing_data = bsf_sharing_data;
        ws_th_data.query_threads = query_threads;
        ws_th_data.pq_th_div_factor = pq_th_div_factor;
        ws_th_data.corr_threshold = args.corr_threshold;
        ws_th_data.verbose = args.verbose;
        ws_th_data.output_file = args.output_file;

        if (pthread_create(&workstealing_thread, nullptr, workstealing_manager, (void *)&ws_th_data) != 0)
        {
            printf("[Node %d]: Error creating workstealing thread\n", my_rank);
            std::exit(EXIT_FAILURE);
        }
    }

    // The scheduler thread serves the other nodes of this rank's replication group, if any.
    const bool scheduler_thread = comm_data != nullptr && comm_data->mode == DynamicSchedulingMode::STANDALONE_THREAD &&
                                  comm_sz > 1 && rep_get_repgroup_nodes(*replication_data, my_rank) > 1;
    if (scheduler_thread)
    {
        CoordinatorData c_data;
        c_data.comm_data = comm_data;
        c_data.threads_finished = &query_workers_finished;
        if (pthread_create(&coordinator_thread_id, nullptr, dynamic_query_scheduler, (void *)&c_data) != 0)
        {
            printf("[Node %d]: Error creating scheduler coordinator thread\n", my_rank);
            std::exit(EXIT_FAILURE);
        }
    }

    query_workers().wait();

    query_workers_finished = 1;

    if (!threshold_mode && comm_sz > 1 && static_cast<int>(workstealing_data->ws_type) > 0)
    {
        if (pthread_join(workstealing_thread, nullptr) != 0)
        {
            printf("[Node %d]: Error joining workstealing thread\n", my_rank);
            std::exit(EXIT_FAILURE);
        }
    }

    if (scheduler_thread)
    {
        if (pthread_join(coordinator_thread_id, nullptr) != 0)
        {
            printf("[Node %d]: Error joining coordinator thread\n", my_rank);
            std::exit(EXIT_FAILURE);
        }
    }

    pthread_barrier_destroy(&sync_barrier);
    pthread_mutex_destroy(&bsf_lock);
    pthread_mutex_destroy(&distances_lock);
    if (knn_exclusion)
    {
        const float bound = bsf_result.pq_bsf->knn[bsf_result.pq_bsf->k - 1];
        if (static_cast<size_t>(query_id) >= g_knn_candidates.size())
            g_knn_candidates.resize(static_cast<size_t>(query_id) + 1);
        std::vector<KnnCandidate> &kept = g_knn_candidates[static_cast<size_t>(query_id)];
        kept.clear();
        for (const KnnCandidate &c : knn_candidates)
        {
            if (c.second <= bound)
                kept.push_back(c);
        }
    }
    if (threshold_mode && threshold_output_file != nullptr)
    {
        std::sort(threshold_hits.begin(), threshold_hits.end(), [](const auto &a, const auto &b)
                  {
                if (a.first != b.first)
                    return a.first < b.first;
                return a.second < b.second; });

        std::vector<std::pair<file_position_type, float>> unique_hits;
        unique_hits.reserve(threshold_hits.size());
        for (const auto &hit : threshold_hits)
        {
            if (unique_hits.empty() || unique_hits.back().first != hit.first)
                unique_hits.push_back(hit);
        }

        bool has_last = false;
        file_position_type last_accepted = 0;
        threshold_match_counter = 0;
        for (const auto &hit : unique_hits)
        {
            const file_position_type pos = hit.first;
            if (has_last && merge_offset > 0 && pos < last_accepted + static_cast<file_position_type>(merge_offset + 1))
                continue;

            std::fprintf(threshold_output_file,
                         "%d, %llu, %llu, %f\n",
                         query_counter,
                         static_cast<unsigned long long>(threshold_match_counter),
                         static_cast<unsigned long long>(pos),
                         ed_to_cc(hit.second, index->settings->timeseries_size));
            threshold_match_counter++;
            last_accepted = pos;
            has_last = true;
        }
    }
    if (threshold_hits_lock_initialized)
    {
        pthread_mutex_destroy(&threshold_hits_lock);
    }
    if (threshold_output_lock_initialized)
    {
        pthread_mutex_destroy(&threshold_output_lock);
    }
    if (threshold_output_file != nullptr && std::ferror(threshold_output_file))
    {
        std::fprintf(stderr, "[Node %d] Error: failed while writing threshold output: %s\n", my_rank, std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    threshold_output_file = nullptr; // stays open for the next query (g_threshold_output)

    if (all_pqs != nullptr)
    {
        for (int i = 0; i < all_pqs_size; i++)
        {
            pqueue_free(all_pqs[i]);
            all_pqs[i] = nullptr;
        }
        free(all_pqs);
    }
    release_subtree_batches(batchlist);

    // Per-query time on this node and whether the index or the scan answered it.
    const bool scanned = g_refinement_mode == RefinementMode::Adaptive && adaptive_decision == 1;
    const bool fft = g_refinement_mode == RefinementMode::Adaptive && adaptive_decision == 2;
    std::printf("[Node %d] query %d: %.6f s (%s)\n", my_rank, query_counter, MPI_Wtime() - query_start,
                fft ? "fft" : scanned ? "scan"
                                      : "index");
    return bsf_result;
}

query_result qa_exact_search_tremor_knn_workstealing(WsSearchFunctionParams ws_args)
{
    int query_id = ws_args.query_id;
    ts_type *ts = ws_args.ts;
    ts_type *paa = ws_args.paa;
    isax_index *index = ws_args.index;
    NodeList *nodelist = ws_args.nodelist;
    float minimum_distance = ws_args.minimum_distance;
    double (*estimation_func)(double) = ws_args.estimation_func;
    float bsf = ws_args.bsf;
    file_position_type bsf_pos = ws_args.bsf_pos;
    int *batches_to_create = ws_args.batch_ids;
    std::vector<BsfMessage> *shared_bsf_results = ws_args.shared_bsf_results;
    int top_k = ws_args.k;
    int merge_offset = ws_args.merge_offset;
    int my_rank = ws_args.my_rank;
    int query_threads = ws_args.query_threads;
    int pq_th_div_factor = ws_args.pq_th_div_factor;
    int comm_sz = ws_args.comm_sz;
    WorkstealingData *workstealing_data = ws_args.workstealing_data;
    ReplicationData *replication_data = ws_args.replication_data;
    float *rawfile = ws_args.rawfile;
    BsfSharingData *bsf_sharing_data = ws_args.bsf_sharing_data;
    ws_args.query_counter = query_id;
    int query_counter = ws_args.query_counter;
    const bool threshold_mode = ws_args.threshold_mode;

    query_result bsf_result;
    bsf_result.distance = FLT_MAX;
    bsf_result.node = nullptr;
    bsf_result.pqueue_position = 0;
    bsf_result.pq_bsf = pqueue_bsf_init_from_val(top_k, bsf, bsf_pos);
    bsf_result.total_time = 0.0f;

    int median = estimate_th(bsf, estimation_func);
    int query_th = median / pq_th_div_factor;
    if (query_th <= 0)
        query_th = 1;

    BatchList *batchlist = create_subtree_batches(nodelist, query_threads, query_th);

    std::vector<pthread_t> threadid(static_cast<size_t>(query_threads));
    pthread_barrier_t sync_barrier;
    pthread_barrier_init(&sync_barrier, nullptr, query_threads);
    pthread_mutex_t bsf_lock;
    pthread_mutex_t distances_lock;
    pthread_mutex_init(&bsf_lock, nullptr);
    pthread_mutex_init(&distances_lock, nullptr);

    volatile int batch_counter = 0;
    volatile int pq_counter = 0;
    volatile char query_workers_finished = 0;
    volatile char receiving_workstealing = 1;
    volatile char priority_queues_filled = 0;

    SubtreeBatch *batches = batchlist->batches;
    int total_batches = batchlist->batch_amount;

    SubtreeBatch *ws_batches = nullptr;
    if (workstealing_data->ws_type == WorkstealingType::S_WS)
    {
        ws_batches = static_cast<SubtreeBatch *>(std::malloc(sizeof(SubtreeBatch) * static_cast<size_t>(workstealing_data->items_to_send)));
        if (ws_batches == nullptr)
        {
            release_subtree_batches(batchlist);
            pqueue_bsf_destroy(bsf_result.pq_bsf);
            bsf_result.pq_bsf = nullptr;
            return bsf_result;
        }
        total_batches = workstealing_data->items_to_send;
        for (int i = 0; i < workstealing_data->items_to_send; i++)
        {
            int batch_id = batches_to_create[i];
            ws_batches[i] = batchlist->batches[batch_id];
        }
        batches = ws_batches;
    }

    pqueue_t **all_pqs = nullptr;
    int all_pqs_size = 0;
    int pqs_processed = 0;
    int pqs_stolen = 0;

    const std::vector<int> block_order = query_block_order(ts, index->settings->timeseries_size);
    std::vector<QaWorkerData> workerdata(static_cast<size_t>(query_threads));
    for (int i = 0; i < query_threads; i++)
    {
        workerdata[static_cast<size_t>(i)].workernumber = i;
        workerdata[static_cast<size_t>(i)].ts = ts;
        workerdata[static_cast<size_t>(i)].block_order = block_order.data();
        // Stolen work is always verified leaf by leaf.
        workerdata[static_cast<size_t>(i)].refine_cursor = nullptr;
        workerdata[static_cast<size_t>(i)].knn_candidates = nullptr;
        workerdata[static_cast<size_t>(i)].adaptive_decision = nullptr;
        workerdata[static_cast<size_t>(i)].adaptive_sample_candidates = nullptr;
        workerdata[static_cast<size_t>(i)].adaptive_sample_blocks = nullptr;
        workerdata[static_cast<size_t>(i)].seed_leaf = nullptr;
        workerdata[static_cast<size_t>(i)].record_lb = true;
        workerdata[static_cast<size_t>(i)].share_seed = false;
        workerdata[static_cast<size_t>(i)].paa = paa;
        workerdata[static_cast<size_t>(i)].minimum_distance = minimum_distance;
        workerdata[static_cast<size_t>(i)].bsf_result = &bsf_result;
        workerdata[static_cast<size_t>(i)].batches = batches;
        workerdata[static_cast<size_t>(i)].total_batches = total_batches;
        workerdata[static_cast<size_t>(i)].batch_counter = &batch_counter;
        workerdata[static_cast<size_t>(i)].sync_barrier = &sync_barrier;
        workerdata[static_cast<size_t>(i)].bsf_lock = &bsf_lock;
        workerdata[static_cast<size_t>(i)].receiving_workstealing = &receiving_workstealing;
        workerdata[static_cast<size_t>(i)].index = index;
        workerdata[static_cast<size_t>(i)].pq_counter = &pq_counter;
        workerdata[static_cast<size_t>(i)].final_pq_list = &all_pqs;
        workerdata[static_cast<size_t>(i)].final_pq_list_size = &all_pqs_size;
        workerdata[static_cast<size_t>(i)].distances_lock = &distances_lock;
        workerdata[static_cast<size_t>(i)].priority_queues_filled = &priority_queues_filled;
        workerdata[static_cast<size_t>(i)].pqs_stolen = &pqs_stolen;
        workerdata[static_cast<size_t>(i)].processed_pqs = &pqs_processed;
        workerdata[static_cast<size_t>(i)].shared_bsf_results = shared_bsf_results;
        workerdata[static_cast<size_t>(i)].comm_data = nullptr;
        workerdata[static_cast<size_t>(i)].query_counter = query_counter;
        workerdata[static_cast<size_t>(i)].replication_data = replication_data;
        workerdata[static_cast<size_t>(i)].workstealing_data = workstealing_data;
        workerdata[static_cast<size_t>(i)].query_threads = query_threads;
        workerdata[static_cast<size_t>(i)].comm_sz = comm_sz;
        workerdata[static_cast<size_t>(i)].my_rank = my_rank;
        workerdata[static_cast<size_t>(i)].merge_offset = merge_offset;
        workerdata[static_cast<size_t>(i)].corr_threshold = ws_args.corr_threshold;
        workerdata[static_cast<size_t>(i)].rawfile = rawfile;
        workerdata[static_cast<size_t>(i)].bsf_sharing_data = bsf_sharing_data;
        workerdata[static_cast<size_t>(i)].pq_th_div_factor = pq_th_div_factor;
        workerdata[static_cast<size_t>(i)].verbose = ws_args.verbose;
        workerdata[static_cast<size_t>(i)].output_file = ws_args.output_file;
        workerdata[static_cast<size_t>(i)].warp_window = ws_args.warp_window;
        workerdata[static_cast<size_t>(i)].paaU = ws_args.paaU;
        workerdata[static_cast<size_t>(i)].paaL = ws_args.paaL;
        workerdata[static_cast<size_t>(i)].local_chunk_start = ws_args.local_chunk_start;
        workerdata[static_cast<size_t>(i)].local_chunk_length = ws_args.local_chunk_length;
        workerdata[static_cast<size_t>(i)].local_owned_subseq_start = ws_args.local_owned_subseq_start;
        workerdata[static_cast<size_t>(i)].local_owned_subseq_count = ws_args.local_owned_subseq_count;
        workerdata[static_cast<size_t>(i)].subsequence_means = ws_args.subsequence_means;
        workerdata[static_cast<size_t>(i)].subsequence_stds = ws_args.subsequence_stds;
        workerdata[static_cast<size_t>(i)].subsequence_length = ws_args.subsequence_length;
        workerdata[static_cast<size_t>(i)].threshold_mode = threshold_mode;
        workerdata[static_cast<size_t>(i)].threshold_ed = ws_args.threshold_ed;
        workerdata[static_cast<size_t>(i)].threshold_output_file = ws_args.threshold_output_file;
        workerdata[static_cast<size_t>(i)].threshold_output_lock = ws_args.threshold_output_lock;
        workerdata[static_cast<size_t>(i)].threshold_match_counter = ws_args.threshold_match_counter;
        workerdata[static_cast<size_t>(i)].threshold_hits = ws_args.threshold_hits;
        workerdata[static_cast<size_t>(i)].threshold_hits_lock = ws_args.threshold_hits_lock;

        if (pthread_create(&threadid[static_cast<size_t>(i)], worker_attr(i), qa_exact_search_tremor_worker, (void *)&workerdata[static_cast<size_t>(i)]) != 0)
        {
            printf("[Node %d]: Error creating thread %d for qa_exact_search_tremor_knn_worker (workstealing)\n", my_rank, i);
            std::exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < query_threads; i++)
    {
        if (pthread_join(threadid[static_cast<size_t>(i)], nullptr) != 0)
        {
            printf("[Node %d]: Error joining qa_exact_search_tremor_knn_worker %d (workstealing)\n", my_rank, i);
            std::exit(EXIT_FAILURE);
        }
    }

    query_workers_finished = 1;

    pthread_barrier_destroy(&sync_barrier);
    pthread_mutex_destroy(&bsf_lock);
    pthread_mutex_destroy(&distances_lock);

    if (all_pqs != nullptr)
    {
        for (int i = 0; i < all_pqs_size; i++)
        {
            pqueue_free(all_pqs[i]);
            all_pqs[i] = nullptr;
        }
        free(all_pqs);
    }

    if (ws_batches != nullptr)
        free(ws_batches);
    release_subtree_batches(batchlist);

    (void)pqs_stolen;
    (void)pqs_processed;
    return bsf_result;
}

void Tremor::initializeMPI(int argc, char **argv)
{
    int already_initialized = 0;
    MPI_Initialized(&already_initialized);
    if (!already_initialized)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE)
        {
            printf("The threading support level is lesser than that demanded.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    MPI_Comm_size(MPI_COMM_WORLD, &this->comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->my_rank);
}

Tremor::Tremor(DistanceType distance_type)
{
    this->distance_type = distance_type;

    int argc = 0;
    char **argv = nullptr;
    initializeMPI(argc, argv);

    if (this->time_series_size % 8 != 0)
    {
        if (this->my_rank == 0)
        {
            std::cerr << "[Node " << this->my_rank
                      << "]: Error: SIMD calculations require query length to be a multiple of 8. Current value is "
                      << this->time_series_size << "\n";
        }
        std::exit(EXIT_FAILURE);
    }
}

Tremor::Tremor(DistanceType distance_type, int argc, char **argv)
{
    this->distance_type = distance_type;
    initializeMPI(argc, argv);

    if (this->time_series_size % 8 != 0)
    {
        if (this->my_rank == 0)
        {
            std::cerr << "[Node " << this->my_rank
                      << "]: Error: SIMD calculations require query length to be a multiple of 8. Current value is "
                      << this->time_series_size << "\n";
        }
        std::exit(EXIT_FAILURE);
    }
}

Tremor::Tremor(const TremorConfig &config, DistanceType distance_type, int argc, char **argv)
{
    this->distance_type = distance_type;
    this->search_workers = config.search_workers;
    this->index_threads = config.index_threads;
    this->leaf_size = config.leaf_size;
    this->paa_segments = config.paa_segments;
    this->replication_groups = config.replication_groups;
    this->query_threads = config.query_threads;
    this->pq_th_div_factor = config.pq_th_div_factor;
    this->num_threads = config.search_workers;

    initializeMPI(argc, argv);

    if (this->time_series_size % 8 != 0)
    {
        if (this->my_rank == 0)
        {
            std::cerr << "[Node " << this->my_rank
                      << "]: Error: SIMD calculations require query length to be a multiple of 8. Current value is "
                      << this->time_series_size << "\n";
        }
        std::exit(EXIT_FAILURE);
    }
}

void Tremor::setNumThreads(int num_threads)
{
    int max_threads = omp_get_max_threads();

    if (num_threads > max_threads)
    {
        std::cerr << "[Warning] " << num_threads
                  << " threads exceeds max available " << max_threads << " Using the max threads available.\n";
        this->num_threads = max_threads;
    }
    else if (num_threads < 1)
    {
        std::cerr << "[Warning] Thread count must be >= 1. Using 1.\n";
        this->num_threads = 1;
    }
    else
    {
        this->num_threads = num_threads;
    }
}

int Tremor::getNumThreads() const
{
    return this->num_threads;
}

void Tremor::buildIndex(float *database, idx_t n_database, idx_t dim)
{
    buildIndexLongSequence(database, n_database, dim);
}

void Tremor::buildIndex(const std::string &filename, idx_t dim, idx_t n_database)
{
    const idx_t long_sequence_length =
        (n_database == 0) ? tremor_float32_count_from_file(filename) : n_database;
    buildIndexLongSequence(filename, long_sequence_length, dim);
}

void Tremor::buildIndexLongSequence(float *database, idx_t long_sequence_length, idx_t subsequence_length)
{
    if (database == nullptr)
    {
        throw std::runtime_error("Tremor::buildIndex requires a non-null in-memory database");
    }

    if (subsequence_length == 0 || long_sequence_length == 0)
    {
        throw std::runtime_error("Tremor::buildIndex received empty input dimensions");
    }

    this->dim = subsequence_length;
    this->time_series_size = static_cast<int>(subsequence_length);
    this->long_sequence_length = long_sequence_length;
    this->dataset_size = this->long_sequence_length;
    this->n_database = (this->long_sequence_length >= static_cast<idx_t>(this->time_series_size))
                           ? (this->long_sequence_length - static_cast<idx_t>(this->time_series_size) + 1)
                           : 0;

    tremor_optimize_params(this);
    tremor_prepare_structures(this);
    tremor_log_parameters(this);
    this->rawfile = this->buildIndexSequence(database, nullptr);

    if (this->verbose)
    {
        print_index_stats(this->index, this->my_rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void Tremor::buildIndexLongSequence(const std::string &filename,
                                    idx_t long_sequence_length,
                                    idx_t subsequence_length)
{
    if (filename.empty())
    {
        throw std::runtime_error("Tremor::buildIndex requires a non-empty dataset filename");
    }

    if (subsequence_length == 0 || long_sequence_length == 0)
    {
        throw std::runtime_error("Tremor::buildIndex received empty input dimensions");
    }

    const idx_t file_floats = tremor_float32_count_from_file(filename);
    if (long_sequence_length > file_floats)
    {
        throw std::runtime_error("Dataset file contains fewer float32 values than requested");
    }

    this->dim = subsequence_length;
    this->time_series_size = static_cast<int>(subsequence_length);
    this->long_sequence_length = long_sequence_length;
    this->dataset_size = this->long_sequence_length;
    this->n_database = (this->long_sequence_length >= static_cast<idx_t>(this->time_series_size))
                           ? (this->long_sequence_length - static_cast<idx_t>(this->time_series_size) + 1)
                           : 0;

    tremor_optimize_params(this);
    tremor_prepare_structures(this);
    tremor_log_parameters(this);
    this->rawfile = this->buildIndexSequence(nullptr, &filename);

    if (this->verbose)
    {
        print_index_stats(this->index, this->my_rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

bool Tremor::validateSearchParams(const idx_t k, const idx_t n_query) const
{
    if (k == 0)
    {
        std::cerr << "[Error] k must be greater than 0\n";
        return false;
    }
    if (mode != 0 && k > n_database)
    {
        std::cerr << "[Error] k (" << k << ") cannot be greater than database size (" << n_database << ")\n";
        return false;
    }
    if (n_query == 0)
    {
        std::cerr << "[Error] n_query must be greater than 0\n";
        return false;
    }
    if (index == nullptr)
    {
        std::cerr << "[Error] [Node " << getMyRank() << "] Index must be built before searching.\n";
        return false;
    }
    return true;
}

void Tremor::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
{
    if (!validateSearchParams(k, n_query))
        return;

    this->top_k = static_cast<int>(k);

    const int q_num = static_cast<int>(n_query);
    const int topk = static_cast<int>(k);
    isax_index *idx = this->index;
    const int my_rank = this->my_rank;
    const int comm_sz = this->comm_sz;

    TremorQuery *queries = load_queries_from_buffer(query, q_num, idx, my_rank);
    g_knn_candidates.assign(static_cast<size_t>(q_num), {});

    double (*basis_func)(double) = initialize_basis_function(dataset_type.c_str());

    tremor_preprocess_and_sort_queries(this, queries, q_num, true);

    query_result *results = (query_result *)malloc(sizeof(query_result) * static_cast<size_t>(q_num));
    if (!results)
    {
        std::cerr << "[Node " << my_rank << "]: Failed to allocate results.\n";
        free_queries(queries, q_num);
        return;
    }

    std::vector<BsfMessage> shared_bsf_results(static_cast<size_t>(q_num));
    for (int i = 0; i < q_num; i++)
    {
        results[i].distance = FLT_MAX;
        results[i].pq_bsf = nullptr;
        results[i].total_time = FLT_MAX;
        shared_bsf_results[i].bsf = FLT_MAX;
        shared_bsf_results[i].position = 0;
        shared_bsf_results[i].q_num = i;
    }

    if (my_rank == 0 && verbose)
    {
        for (int i = 0; i < q_num; i++)
        {
            if (queries[i].initial_pq_bsfs != nullptr)
                printf("[Node %d]: Query %d, Actual ID: %d, BSF: %f\n", my_rank, i, queries[i].id,
                       queries[i].initial_pq_bsfs->knn[topk - 1]);
        }
    }

    NodeList nodelist = initialize_node_list(idx, my_rank);
    searchIndexL2Squared(queries, q_num, topk, results, &shared_bsf_results, nodelist, I, D, basis_func);

    // Free per-query BSF queues before freeing the results array to
    // avoid leaking the heap allocations made inside
    // qa_exact_search_tremor_knn / pqueue_bsf_init*.
    for (int i = 0; i < q_num; i++)
    {
        if (results[i].pq_bsf != nullptr)
        {
            pqueue_bsf_destroy(results[i].pq_bsf);
            results[i].pq_bsf = nullptr;
        }
    }
    free(results);
}

void Tremor::searchIndexL2Squared(TremorQuery *queries, int q_num, int topk,
                                  query_result *results, std::vector<BsfMessage> *shared_bsf_results,
                                  NodeList &nodelist, idx_t *I, float *D,
                                  double (*basis_func)(double))
{
    constexpr bool ENABLE_PRINTS_PER_QUERY = false;
    const float minimum_distance = FLT_MAX;
    isax_index *index = this->index;
    ReplicationData &replication_data = this->replication_data;
    WorkstealingData *workstealing_data = &this->workstealing_data;
    const DynamicSchedulingMode mode = static_cast<DynamicSchedulingMode>(this->dynamic_scheduling_mode);
    const int num_procs = this->comm_sz;
    const int rank = this->my_rank;

    if (this->mode == 0)
    {
        tremor_prepare_threshold_output_file(output_file, rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (num_procs == 1)
    {
        for (int q_loaded = 0; q_loaded < q_num; q_loaded++)
        {
            SearchFunctionParams args;
            args.query_id = q_loaded;
            args.ts = queries[q_loaded].query;
            args.paa = queries[q_loaded].paa;
            args.index = index;
            args.nodelist = &nodelist;
            args.minimum_distance = minimum_distance;
            args.comm_data = nullptr;
            args.estimation_func = basis_func;
            args.shared_bsf_results = shared_bsf_results;
            args.k = topk;
            args.precomputed_bsfs = queries[q_loaded].initial_pq_bsfs;
            args.query_threads = query_threads;
            args.my_rank = my_rank;
            args.comm_sz = comm_sz;
            args.verbose = verbose;
            args.rawfile = rawfile;
            args.replication_data = &replication_data;
            args.output_file = output_file;
            args.corr_threshold = corr_threshold;
            args.bsf_sharing_data = &bsf_sharing_data;
            args.workstealing_data = workstealing_data;
            args.pq_th_div_factor = pq_th_div_factor;
            args.merge_offset = merge_offset;
            args.query_counter = q_loaded;
            args.warp_window = 0;
            args.paaU = nullptr;
            args.paaL = nullptr;
            args.local_chunk_start = local_chunk_start;
            args.local_chunk_length = local_chunk_length;
            args.local_owned_subseq_start = local_owned_subseq_start;
            args.local_owned_subseq_count = local_owned_subseq_count;
            args.subsequence_means = subsequence_means;
            args.subsequence_stds = subsequence_stds;
            args.subsequence_length = time_series_size;
            args.threshold_mode = (this->mode == 0);
            args.threshold_ed = cc_to_ed(corr_threshold, time_series_size);
            args.threshold_output_file = nullptr;
            args.threshold_output_lock = nullptr;
            args.threshold_match_counter = nullptr;
            args.threshold_hits = nullptr;
            args.threshold_hits_lock = nullptr;

            query_result result = qa_exact_search_tremor_knn(args);
            results[queries[q_loaded].id] = result;
        }
    }
    else
    {
        std::vector<MPI_Request> request(static_cast<size_t>(comm_sz));
        std::vector<MPI_Request> send_request(static_cast<size_t>(comm_sz));
        const int distributed_queries_initial_burst = 1;
        std::vector<int *> process_buffer_initial(static_cast<size_t>(comm_sz));
        for (int i = 0; i < comm_sz; i++)
        {
            process_buffer_initial[static_cast<size_t>(i)] = (int *)malloc(sizeof(int) * static_cast<size_t>(distributed_queries_initial_burst));
            CHECK_ALLOC(process_buffer_initial[static_cast<size_t>(i)], my_rank);
        }
        std::vector<int> process_buffer(static_cast<size_t>(comm_sz), 0);
        int rec_message = 0;
        int buffer_sent = 0;
        int termination_message_id = -1;
        int q_loaded = 0;

        const int DISTRIBUTED_QUERIES_SEND_QUERY = 800;
        const int DISTRIBUTED_QUERIES_REQUEST_QUERY = 801;

        if (my_rank == rep_find_coordinator_node_rank(replication_data, my_rank))
        {
            send_initial_queries_module_coordinator_async_chatzakis(&q_loaded, my_rank, comm_sz,
                                                                    distributed_queries_initial_burst,
                                                                    process_buffer_initial.data(),
                                                                    &rec_message, request.data(), send_request.data(),
                                                                    q_num, &termination_message_id,
                                                                    &replication_data);
        }

        while (1)
        {
            if (my_rank == rep_find_coordinator_node_rank(replication_data, my_rank))
            {
                if (mode == DynamicSchedulingMode::PERIODIC_CHECK || mode == DynamicSchedulingMode::STANDALONE_THREAD)
                {
                    if (q_loaded >= q_num || q_loaded == -1)
                    {
                        if (verbose)
                            printf("[Node %d]: Exiting the dynamic loop\n", my_rank);
                        break;
                    }

                    int query_to_keep_stats = q_loaded;
                    q_loaded++;

                    CommunicationModuleData comm_data;
                    comm_data.module_func = &send_queries_module_coordinator_async_chatzakis;
                    comm_data.q_loaded = &q_loaded;
                    comm_data.rec_message = &rec_message;
                    comm_data.termination_message_id = &termination_message_id;
                    comm_data.q_num = q_num;
                    comm_data.request = request.data();
                    comm_data.send_request = send_request.data();
                    comm_data.process_buffer = process_buffer.data();
                    comm_data.mode = mode;
                    comm_data.my_rank = my_rank;
                    comm_data.comm_sz = comm_sz;
                    comm_data.replication_data = &replication_data;
                    comm_data.verbose = verbose;

                    SearchFunctionParams args;
                    args.query_id = query_to_keep_stats;
                    args.ts = queries[query_to_keep_stats].query;
                    args.paa = queries[query_to_keep_stats].paa;
                    args.index = index;
                    args.nodelist = &nodelist;
                    args.minimum_distance = minimum_distance;
                    args.comm_data = &comm_data;
                    args.estimation_func = basis_func;
                    args.shared_bsf_results = shared_bsf_results;
                    args.k = topk;
                    args.precomputed_bsfs = queries[query_to_keep_stats].initial_pq_bsfs;
                    args.query_threads = query_threads;
                    args.my_rank = my_rank;
                    args.comm_sz = comm_sz;
                    args.verbose = verbose;
                    args.rawfile = rawfile;
                    args.replication_data = &replication_data;
                    args.output_file = output_file;
                    args.corr_threshold = corr_threshold;
                    args.bsf_sharing_data = &bsf_sharing_data;
                    args.workstealing_data = workstealing_data;
                    args.pq_th_div_factor = pq_th_div_factor;
                    args.merge_offset = merge_offset;
                    args.query_counter = query_to_keep_stats;
                    args.warp_window = 0;
                    args.paaU = nullptr;
                    args.paaL = nullptr;
                    args.local_chunk_start = local_chunk_start;
                    args.local_chunk_length = local_chunk_length;
                    args.local_owned_subseq_start = local_owned_subseq_start;
                    args.local_owned_subseq_count = local_owned_subseq_count;
                    args.subsequence_means = subsequence_means;
                    args.subsequence_stds = subsequence_stds;
                    args.subsequence_length = time_series_size;
                    args.threshold_mode = (this->mode == 0);
                    args.threshold_ed = cc_to_ed(corr_threshold, time_series_size);
                    args.threshold_output_file = nullptr;
                    args.threshold_output_lock = nullptr;
                    args.threshold_match_counter = nullptr;
                    args.threshold_hits = nullptr;
                    args.threshold_hits_lock = nullptr;

                    query_result result = qa_exact_search_tremor_knn(args);
                    result.total_time = 0.0;

                    int query_id = queries[query_to_keep_stats].id;
                    results[query_id] = result;

                    if (ENABLE_PRINTS_PER_QUERY && verbose && results[query_id].pq_bsf != nullptr)
                    {
                        printf("[Node %d]: Processed query %d (id %d) => (1-nn=%f, pos=%llu)\n",
                               my_rank, query_to_keep_stats, query_id,
                               results[query_id].pq_bsf->knn[0],
                               (unsigned long long)results[query_id].pq_bsf->position[0]);
                    }
                }

                if (!send_queries_module_coordinator_async_chatzakis(&q_loaded, q_num, process_buffer.data(),
                                                                     request.data(), &rec_message, send_request.data(),
                                                                     &termination_message_id,
                                                                     &replication_data, my_rank, comm_sz, verbose))
                {
                    break;
                }
            }
            else
            {
                MPI_Recv(&q_loaded, 1, MPI_INT, rep_find_coordinator_node_rank(replication_data, my_rank),
                         DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (q_loaded >= q_num || q_loaded == -1)
                {
                    break;
                }

                SearchFunctionParams args;
                args.query_id = q_loaded;
                args.ts = queries[q_loaded].query;
                args.paa = queries[q_loaded].paa;
                args.index = index;
                args.nodelist = &nodelist;
                args.minimum_distance = minimum_distance;
                args.comm_data = nullptr;
                args.estimation_func = basis_func;
                args.shared_bsf_results = shared_bsf_results;
                args.k = topk;
                args.precomputed_bsfs = queries[q_loaded].initial_pq_bsfs;
                args.query_threads = query_threads;
                args.my_rank = my_rank;
                args.comm_sz = comm_sz;
                args.verbose = verbose;
                args.rawfile = rawfile;
                args.replication_data = &replication_data;
                args.output_file = output_file;
                args.corr_threshold = corr_threshold;
                args.bsf_sharing_data = &bsf_sharing_data;
                args.workstealing_data = workstealing_data;
                args.pq_th_div_factor = pq_th_div_factor;
                args.merge_offset = merge_offset;
                args.query_counter = q_loaded;
                args.warp_window = 0;
                args.paaU = nullptr;
                args.paaL = nullptr;
                args.local_chunk_start = local_chunk_start;
                args.local_chunk_length = local_chunk_length;
                args.local_owned_subseq_start = local_owned_subseq_start;
                args.local_owned_subseq_count = local_owned_subseq_count;
                args.subsequence_means = subsequence_means;
                args.subsequence_stds = subsequence_stds;
                args.subsequence_length = time_series_size;
                args.threshold_mode = (this->mode == 0);
                args.threshold_ed = cc_to_ed(corr_threshold, time_series_size);
                args.threshold_output_file = nullptr;
                args.threshold_output_lock = nullptr;
                args.threshold_match_counter = nullptr;
                args.threshold_hits = nullptr;
                args.threshold_hits_lock = nullptr;

                query_result result = qa_exact_search_tremor_knn(args);
                result.total_time = 0.0;

                int query_id = queries[q_loaded].id;
                results[query_id] = result;

                if (ENABLE_PRINTS_PER_QUERY && verbose && results[query_id].pq_bsf != nullptr)
                {
                    printf("[Node %d]: Processed query %d (id %d) => (1-nn=%f, pos=%llu)\n",
                           my_rank, q_loaded, query_id,
                           results[query_id].pq_bsf->knn[0],
                           (unsigned long long)results[query_id].pq_bsf->position[0]);
                }

                if (q_loaded == q_num - 1)
                {
                    break;
                }

                MPI_Isend(&buffer_sent, 1, MPI_INT, rep_find_coordinator_node_rank(replication_data, my_rank),
                          DISTRIBUTED_QUERIES_REQUEST_QUERY, MPI_COMM_WORLD,
                          &send_request[static_cast<size_t>(rep_find_coordinator_node_rank(replication_data, my_rank))]);
            }
        }

        for (size_t i = 0; i < static_cast<size_t>(comm_sz); i++)
            free(process_buffer_initial[i]);
    }

    if (this->mode != 0 && num_procs > 1 && workstealing_data->ws_type != WorkstealingType::DISABLED)
    {
        tremor_perform_workstealing(this, queries, nodelist,
                                    &qa_exact_search_tremor_knn_workstealing,
                                    basis_func, results, shared_bsf_results);
    }

#if defined(TREMOR_DEBUG_RESULTS) && TREMOR_DEBUG_RESULTS
    for (int i = 0; i < q_num; i++)
    {
        if (results[i].pq_bsf != nullptr)
        {
            printf("[Node %d] DEBUG results[%d]: ", my_rank, i);
            for (int j = 0; j < topk; j++)
                printf("(pos=%llu dist=%.4f) ", (unsigned long long)results[i].pq_bsf->position[j], results[i].pq_bsf->knn[j]);
            printf("\n");
        }
        else
            printf("[Node %d] DEBUG results[%d]: pq_bsf is NULL\n", my_rank, i);
    }
    fflush(stdout);
#endif

    const size_t result_count = static_cast<size_t>(q_num) * static_cast<size_t>(topk);
    std::memset(I, 0, result_count * sizeof(idx_t));
    std::fill(D, D + result_count, FLT_MAX);

    if (this->mode == 0)
    {
        if (g_threshold_output != nullptr &&
            (std::fflush(g_threshold_output) != 0 || std::ferror(g_threshold_output) || std::fclose(g_threshold_output) != 0))
        {
            std::fprintf(stderr, "[Node %d] Error: failed to write threshold output: %s\n", my_rank, std::strerror(errno));
            std::exit(EXIT_FAILURE);
        }
        g_threshold_output = nullptr;
        free(nodelist.nlist);
        free_queries(queries, q_num);
        return;
    }

    for (int i = 0; i < q_num; i++)
    {
        if (results[i].pq_bsf != nullptr)
        {
            if (num_procs == 1)
            {

                std::vector<std::pair<float, idx_t>> pairs;
                pairs.reserve(static_cast<size_t>(topk));
                for (int j = 0; j < topk; j++)
                {
                    const long pos_signed = results[i].pq_bsf->position[j];
                    const float dist = results[i].pq_bsf->knn[j];
                    if (pos_signed >= 0 && dist < FLT_MAX * 0.99f)
                        pairs.emplace_back(dist, static_cast<idx_t>(pos_signed));
                }
                std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b)
                          {
                        if (a.first != b.first) return a.first < b.first;
                        return a.second < b.second; });
                std::unordered_set<idx_t> seen_pos;
                std::vector<std::pair<float, idx_t>> uniq;
                uniq.reserve(pairs.size());
                for (const auto &p : pairs)
                {
                    if (seen_pos.insert(p.second).second)
                        uniq.push_back(p);
                }
                // Leave empty slots as (0, FLT_MAX) instead of
                // replicating the last valid entry. This matches DMASS
                // and keeps spurious duplicates out of the output for
                // queries with fewer than topk valid hits.
                for (int j = 0; j < topk; j++)
                {
                    idx_t out_pos = 0;
                    float out_dist = FLT_MAX;
                    if (j < static_cast<int>(uniq.size()))
                    {
                        out_dist = uniq[static_cast<size_t>(j)].first;
                        out_pos = uniq[static_cast<size_t>(j)].second;
                    }
                    I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = out_pos;
                    D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = out_dist;
                }
            }
            else
            {
                for (int j = 0; j < topk; j++)
                {
                    const long pos_signed = results[i].pq_bsf->position[j];
                    idx_t pos = static_cast<idx_t>(pos_signed);
                    I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = pos;
                    D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = results[i].pq_bsf->knn[j];
                }
            }
        }
        else if (num_procs > 1)
        {

            for (int j = 0; j < topk; j++)
            {
                I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = 0;
                D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = FLT_MAX;
            }
        }
    }

    if (this->merge_offset > 0)
    {
        tremor_knn_exclusion_answers(rank, num_procs, q_num, topk, this->merge_offset, queries, I, D);
    }
    else if (num_procs > 1)
    {
        tremor_merge_knn_results_mpi(rank, num_procs, q_num, topk, I, D, this->merge_offset);
    }

    tremor_write_knn_output_if_requested(output_file, rank, q_num, topk, I, D, time_series_size);

    free(nodelist.nlist);
    free_queries(queries, q_num);
}

float *Tremor::buildIndexSequence(const float *database, const std::string *filename)
{

    isax_index *index = this->index;
    int my_rank = this->my_rank;
    ReplicationData *replication_data = &this->replication_data;
    int index_threads = this->index_threads;

    if (database == nullptr && filename == nullptr)
    {
        fprintf(stderr, "[Node %d] Error: no in-memory database or dataset filename provided.\n", my_rank);
        std::exit(EXIT_FAILURE);
    }

    idx_t total_samples = this->dataset_size;
    idx_t my_owned_points = rep_get_time_series_of_group(*replication_data, my_rank);
    idx_t my_owned_start = rep_get_time_series_offset(*replication_data, my_rank);
    const idx_t overlap = (index->settings->timeseries_size > 0)
                              ? static_cast<idx_t>(index->settings->timeseries_size - 1)
                              : 0;

    idx_t my_chunk_points = my_owned_points;
    const int my_group_id = rep_find_group(*replication_data, my_rank);
    const bool is_last_replication_group =
        (my_group_id == replication_data->total_groups - 1);

    if (!is_last_replication_group)
    {
        const idx_t remaining = (my_owned_start + my_owned_points < total_samples)
                                    ? (total_samples - (my_owned_start + my_owned_points))
                                    : 0;
        my_chunk_points += std::min(overlap, remaining);
    }

    if (my_owned_start + my_chunk_points > total_samples)
        my_chunk_points = total_samples - my_owned_start;

    if (database != nullptr && this->long_sequence_length < total_samples)
    {
        fprintf(stderr,
                "[Node %d] In-memory database has only %llu values (expected at least %llu)!\n",
                my_rank,
                static_cast<unsigned long long>(this->long_sequence_length),
                static_cast<unsigned long long>(total_samples));
        std::exit(EXIT_FAILURE);
    }

    float *rawfile = nullptr;
    if (my_chunk_points > 0)
    {
        rawfile = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(my_chunk_points)));
        if (rawfile == nullptr)
        {
            fprintf(stderr, "[Node %d] Error: Memory allocation failed for rawfile.\n", my_rank);
            std::exit(EXIT_FAILURE);
        }
    }

    if (database != nullptr)
    {
        if (my_chunk_points > 0)
        {
            std::memcpy(rawfile,
                        database + static_cast<size_t>(my_owned_start),
                        sizeof(ts_type) * static_cast<size_t>(my_chunk_points));
        }
    }
    else
    {
        try
        {
            tremor_read_float32_chunk(*filename, my_owned_start, my_chunk_points, rawfile);
        }
        catch (const std::exception &ex)
        {
            fprintf(stderr,
                    "[Node %d] Error while reading local dataset chunk: %s\n",
                    my_rank,
                    ex.what());
            std::free(rawfile);
            std::exit(EXIT_FAILURE);
        }
    }

    const idx_t total_subsequences = (total_samples >= static_cast<idx_t>(index->settings->timeseries_size))
                                         ? (total_samples - static_cast<idx_t>(index->settings->timeseries_size) + 1)
                                         : 0;
    const idx_t group_start = rep_get_time_series_offset(*replication_data, my_rank);
    const idx_t group_points = rep_get_time_series_of_group(*replication_data, my_rank);
    const idx_t my_owned_subseq_start = group_start;
    const idx_t my_owned_subseq_end = std::min(group_start + group_points, total_subsequences);
    const idx_t my_owned_subseq_count = (my_owned_subseq_end > my_owned_subseq_start)
                                            ? (my_owned_subseq_end - my_owned_subseq_start)
                                            : 0;

    this->local_chunk_start = my_owned_start;
    this->local_chunk_length = my_chunk_points;
    this->local_owned_subseq_start = my_owned_subseq_start;
    this->local_owned_subseq_count = my_owned_subseq_count;

    if (this->subsequence_means)
    {
        std::free(this->subsequence_means);
        this->subsequence_means = nullptr;
    }
    if (this->subsequence_stds)
    {
        std::free(this->subsequence_stds);
        this->subsequence_stds = nullptr;
    }

    if (my_owned_subseq_count > 0)
    {
        this->subsequence_means = static_cast<float *>(
            std::malloc(sizeof(float) * static_cast<size_t>(my_owned_subseq_count)));
        this->subsequence_stds = static_cast<float *>(
            std::malloc(sizeof(float) * static_cast<size_t>(my_owned_subseq_count)));
        if (this->subsequence_means == nullptr || this->subsequence_stds == nullptr)
        {
            fprintf(stderr, "[Node %d] Error: Memory allocation failed for subsequence stats.\n", my_rank);
            std::exit(EXIT_FAILURE);
        }
    }

    // Index memory = resident growth from here (raw data loaded) to the end of the build.
    const double resident_after_load = resident_bytes();
    g_subsequence_sax.assign(static_cast<size_t>(my_owned_subseq_count) * index->settings->paa_segments, 0);
    printf("[Node %d]: Loaded %llu points from global offset %llu (owned subsequences: %llu).\n",
           my_rank,
           static_cast<unsigned long long>(my_chunk_points),
           static_cast<unsigned long long>(my_owned_start),
           static_cast<unsigned long long>(my_owned_subseq_count));

    index->fbl = reinterpret_cast<first_buffer_layer *>(
        initialize_pRecBuf_ekosmas(
            index->settings->initial_fbl_buffer_size,
            static_cast<int>(std::pow(2.0, index->settings->paa_segments)),
            index->settings->max_total_buffer_size +
                DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1),
            index,
            index_threads));

    std::vector<pthread_t> threadid(static_cast<size_t>(index_threads));

    buffer_data_inmemory_ekosmas *input_data =
        static_cast<buffer_data_inmemory_ekosmas *>(
            std::malloc(sizeof(buffer_data_inmemory_ekosmas) *
                        static_cast<size_t>(index_threads)));
    if (input_data == nullptr)
    {
        fprintf(stderr,
                "[Node %d] Error: Memory allocation failed for buffer_data_inmemory_ekosmas.\n",
                my_rank);
        std::exit(EXIT_FAILURE);
    }

    unsigned long next_block_to_process = 0;
    int node_counter = 0;

    volatile unsigned long *next_iSAX_group =
        static_cast<volatile unsigned long *>(
            std::calloc(static_cast<size_t>(index->fbl->max_total_size),
                        sizeof(unsigned long)));
    if (next_iSAX_group == nullptr)
    {
        fprintf(stderr,
                "[Node %d] Error: Memory allocation failed for next_iSAX_group.\n",
                my_rank);
        std::free(input_data);
        std::exit(EXIT_FAILURE);
    }

    pthread_barrier_t wait_summaries_to_compute;
    pthread_barrier_init(&wait_summaries_to_compute, nullptr,
                         static_cast<unsigned int>(index_threads));

    pthread_mutex_t lock_firstnode = PTHREAD_MUTEX_INITIALIZER;

    for (int i = 0; i < index_threads; i++)
    {
        buffer_data_inmemory_ekosmas &data = input_data[i];
        data.index = index;
        data.lock_firstnode = &lock_firstnode;
        data.workernumber = i;
        data.shared_start_number = &next_block_to_process;
        data.ts_num = my_owned_subseq_count;
        data.wait_summaries_to_compute = &wait_summaries_to_compute;
        data.node_counter = &node_counter;
        data.parallelism_in_subtree = NO_PARALLELISM_IN_SUBTREE;
        data.next_iSAX_group = next_iSAX_group;
        data.rawfile = rawfile;
        data.local_chunk_start = my_owned_start;
        data.local_chunk_length = my_chunk_points;
        data.local_owned_subseq_start = my_owned_subseq_start;
        data.local_owned_subseq_count = my_owned_subseq_count;
        data.subsequence_length = index->settings->timeseries_size;
        data.subsequence_means = this->subsequence_means;
        data.subsequence_stds = this->subsequence_stds;
        data.subsequence_sax = g_subsequence_sax.data();
        data.deterministic_index = this->workstealing_data.deterministic_index;
        data.index_threads = index_threads;
        data.readblock = this->read_block_length;
        data.my_rank = my_rank;
        data.comm_sz = this->comm_sz;
        data.replication_data = replication_data;
    }

    for (int i = 0; i < index_threads; i++)
    {
        if (pthread_create(&threadid[static_cast<size_t>(i)],
                           nullptr,
                           index_creation_sequence_worker,
                           static_cast<void *>(&input_data[i])) != 0)
        {
            fprintf(stderr,
                    "[Node %d] Error: could not create index_creation_sequence_worker thread %d\n",
                    my_rank,
                    i);
            std::free(input_data);
            std::exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < index_threads; i++)
    {
        if (pthread_join(threadid[static_cast<size_t>(i)], nullptr) != 0)
        {
            fprintf(stderr,
                    "[Node %d] Error: could not join index_creation_sequence_worker thread %d\n",
                    my_rank,
                    i);
            std::free(input_data);
            std::exit(EXIT_FAILURE);
        }
    }

    std::free(input_data);
    std::free(const_cast<unsigned long *>(next_iSAX_group));

    // FFT fallback: the spectra of the pieces of the owned windows (fft_fallback.hpp).
    tremor_fft::build(rawfile, my_chunk_points, my_owned_subseq_start - my_owned_start, my_owned_subseq_count,
                      index->settings->timeseries_size, this->subsequence_means, this->subsequence_stds, index_threads,
                      my_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    this->index_memory_bytes = resident_bytes() - resident_after_load;
    return rawfile;
}

Tremor::~Tremor()
{
    if (rawfile)
    {
        std::free(rawfile);
        rawfile = nullptr;
    }
    if (subsequence_means)
    {
        std::free(subsequence_means);
        subsequence_means = nullptr;
    }
    if (subsequence_stds)
    {
        std::free(subsequence_stds);
        subsequence_stds = nullptr;
    }
    delete[] database;
}

void tremor_optimize_params(Tremor *tremor)
{

    if (tremor->comm_sz == 1)
    {

        if (tremor->bsf_sharing_data.bsf_sharing_enabled)
        {
            tremor->bsf_sharing_data.bsf_sharing_enabled = false;
            if (tremor->my_rank == 0)
            {
                printf("[Node %d, OptParams]: Single node execution. Disabling BSF-sharing\n",
                       tremor->my_rank);
            }
        }

        if (tremor->workstealing_data.ws_type != WorkstealingType::DISABLED)
        {
            tremor->workstealing_data.ws_type = WorkstealingType::DISABLED;
            if (tremor->my_rank == 0)
            {
                printf("[Node %d, OptParams]: Single node execution. Disabling Workstealing\n",
                       tremor->my_rank);
            }
        }

        if (tremor->replication_data.total_groups != 1)
        {
            tremor->replication_data.total_groups = 1;
            if (tremor->my_rank == 0)
            {
                printf("[Node %d, OptParams]: Single node execution. Replication groups set to 1\n",
                       tremor->my_rank);
            }
        }

        return;
    }

    if (tremor->replication_data.total_groups == 1)
    {

        if (tremor->bsf_sharing_data.bsf_sharing_enabled)
        {
            tremor->bsf_sharing_data.bsf_sharing_enabled = false;
            if (tremor->my_rank == 0)
            {
                printf("[Node %d, OptParams]: Full replication selected. Disabling BSF-sharing\n",
                       tremor->my_rank);
            }
        }
    }

    if (tremor->replication_data.total_groups == tremor->comm_sz)
    {

        if (tremor->workstealing_data.ws_type != WorkstealingType::DISABLED)
        {
            tremor->workstealing_data.ws_type = WorkstealingType::DISABLED;
            if (tremor->my_rank == 0)
            {
                printf("[Node %d, OptParams]: No replication selected. Disabling Workstealing\n",
                       tremor->my_rank);
            }
        }
    }
}

void tremor_prepare_structures(Tremor *tremor)
{

    tremor->index_settings = isax_index_settings_init(
        "",
        tremor->time_series_size,
        tremor->paa_segments,
        tremor->sax_cardinality,
        tremor->leaf_size,
        tremor->min_leaf_size,
        std::max(tremor->initial_lbl_size, tremor->leaf_size), // leaf buffers at least as big as the leaves
        tremor->flush_limit,
        tremor->initial_fbl_size,
        1,
        0,
        0,
        1,
        1);

    tremor->index = isax_index_init_inmemory_ekosmas(tremor->index_settings);

    bsf_sharing_init(tremor->bsf_sharing_data, tremor->my_rank, tremor->comm_sz);

    if (tremor->replication_groups == 0)
    {

        tremor->replication_data.total_groups = tremor->comm_sz;
    }
    else
    {
        tremor->replication_data.total_groups = tremor->replication_groups;
    }

    rep_init(tremor->replication_data, tremor->dataset_size, tremor->my_rank,
             tremor->comm_sz, tremor->index_threads, tremor->query_threads);

    if (tremor->my_rank == 0)
    {
        printf("[Node 0] Replication Groups Configuration:\n");
        printf("  - Total Replication Groups: %d\n", tremor->replication_data.total_groups);
        printf("  - Total MPI Processes: %d\n", tremor->comm_sz);
        for (int i = 0; i < tremor->replication_data.total_groups; i++)
        {
            printf("  - Group[%d]: %llu time series, %d nodes\n",
                   i,
                   static_cast<unsigned long long>(tremor->replication_data.node_groups[i].total_time_series),
                   tremor->replication_data.node_groups[i].total_nodes);
        }
    }

    ws_init(tremor->workstealing_data, tremor->comm_sz);
}

void tremor_log_parameters(Tremor *tremor)
{
    if (tremor->my_rank == 0 && tremor->verbose)
    {
        const char *scheduling_methods[] = {"Single Node", "Static", "Round Robin", "Dynamic"};
        const char *tremor_modes[] = {"Threshold Search", "k-NN Search"};
        const char *dynamic_scheduling_modes[] = {"Periodic Check", "Standalone Thread"};
        const char *dis_enab[] = {"Disabled", "Enabled"};
        const char *workstealing_types[] = {"Disabled", "S-WS"};

        printf("================ Tremor Settings ================\n");
        printf("Total Processes: [%d]\n", tremor->comm_sz);

        printf("Dataset, Size: [%s, %llu]\n",
               "InMemoryDataSource",
               (unsigned long long)tremor->dataset_size);
        printf("Queries, Size: [%s, %llu]\n",
               "Passed to searchIndex()",
               0ULL);

        printf("PAA Segments, SAX Cardinality: [%d, %d]\n",
               tremor->index->settings->paa_segments,
               tremor->index->settings->sax_bit_cardinality);
        printf("Time-series Size: [%d]\n", tremor->index->settings->timeseries_size);
        printf("Leaf Size, Min Leaf Size, Read Block, Flush Limit: [%d, %d, %d, %d]\n",
               tremor->index->settings->max_leaf_size,
               tremor->index->settings->min_leaf_size,
               tremor->read_block_length,
               tremor->index->settings->max_total_full_buffer_size);

        int mode_idx = tremor->mode;
        int scheduling_idx = tremor->query_scheduling;

        if (mode_idx < 0 || mode_idx >= 2)
            mode_idx = 1;
        if (scheduling_idx < 0 || scheduling_idx >= 4)
            scheduling_idx = 3;

        printf("Mode, Scheduling, Method: [%s, %s, KNN]\n",
               tremor_modes[mode_idx],
               scheduling_methods[scheduling_idx]);

        if (tremor->query_scheduling == 3)
        {
            int dyn_sched_idx = tremor->dynamic_scheduling_mode;
            if (dyn_sched_idx < 0 || dyn_sched_idx >= 2)
                dyn_sched_idx = 1;
            printf("Dynamic Scheduling: [%s]\n", dynamic_scheduling_modes[dyn_sched_idx]);
        }

        if (tremor->mode == 0)
        {
            printf("Merge Offset: [%d]\n", tremor->merge_offset);
        }

        printf("TH Division Factor: [%d]\n", tremor->pq_th_div_factor);
        printf("Dataset-Type: [%s]\n", tremor->dataset_type.c_str());

        int ws_type_idx = static_cast<int>(tremor->workstealing_data.ws_type);
        if (ws_type_idx < 0 || ws_type_idx >= 2)
            ws_type_idx = 0;
        printf("Workstealing: [%s]\n", workstealing_types[ws_type_idx]);

        printf("BSF-Sharing: [%s]\n",
               dis_enab[tremor->bsf_sharing_data.bsf_sharing_enabled ? 1 : 0]);
        printf("Density-Aware Distribution: [%s]\n",
               dis_enab[tremor->density_aware_prepro ? 1 : 0]);
        printf("Output File Name: [%s]\n",
               tremor->output_file.empty() ? "Not Provided" : tremor->output_file.c_str());

        printf("Online znorm: [%s]\n", "Enabled for indexed subsequences");

        printf("KNN k-size: [%d]\n", tremor->top_k);
        printf("Verbose: [%s]\n", tremor->verbose ? "Enabled" : "Disabled");

        rep_log_info(tremor->replication_data, tremor->index_threads, tremor->query_threads);

        printf("==================================================\n");
    }
}
