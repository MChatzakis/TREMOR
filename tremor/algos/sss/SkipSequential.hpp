#ifndef SKIP_SEQUENTIAL_HPP
#define SKIP_SEQUENTIAL_HPP

// Skip-sequential scan baseline (no index tree).
//
// Every subsequence of a node's part of the waveform keeps the same iSAX
// summary as in TREMOR's index, plus its mean and standard deviation. A
// template is answered by scanning the subsequences in waveform order: a
// subsequence is skipped when its iSAX lower bound exceeds the current bound
// (the threshold distance, or the k-th best distance), and otherwise its
// distance is computed with TREMOR's early-abandoning kernel. As in TREMOR, the
// ranks form replication groups of consecutive ranks, the waveform is split
// evenly across the groups, and every rank of a group holds the group's part;
// each template is answered by one rank of every group, which claims it from a
// counter shared by the group (dynamic scheduling), and the results are
// combined as in TREMOR.
//
// Without lower bounds, the same class is the sequential scan baseline (SS):
// no iSAX summaries, and the distance of every subsequence is computed with the
// early-abandoning kernel (TREMOR's last fallback, without its index).

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <mpi.h>

#include "../../distance_computers/ZnormSearch.hpp"
#include "../../isax/iSAXTypes.hpp"

class SkipSequentialScan
{
public:
    // replication_groups: 0 = one group per rank (no replication), 1 = full replication.
    // lower_bounds = false: sequential scan without iSAX summaries and lower bounds.
    SkipSequentialScan(int threads, int paa_segments, int merge_offset, int replication_groups,
                       bool lower_bounds = true);
    ~SkipSequentialScan();

    // Loads this rank's part of the waveform and computes the summaries.
    void build(const std::string &filename, idx_t total_samples, int window_length);

    // Writes <output>_<rank>.csv: "Query, Counter, Position, Cross Correlation".
    void search_threshold(const float *queries, int query_count, float correlation, const std::string &output);

    // Writes <output> on rank 0: "tID, k, pos, corr".
    void search_knn(const float *queries, int query_count, int k, const std::string &output);

    // Resident memory added by build() beyond the raw data (summaries and statistics).
    double summary_memory_bytes() const { return summary_bytes; }

private:
    // Lower bound between a query PAA and the iSAX summary of subsequence `id`.
    float lower_bound(ts_type *query_paa, idx_t id) const;
    // Next template for this rank, or query_count when the group has claimed all.
    int next_query(int query_count);
    // Z-normalized query, its PAA, and the block order of the distance kernel.
    void prepare_query(const float *query, std::vector<ts_type> &znorm, std::vector<ts_type> &paa,
                       std::vector<int> &order) const;

    int threads, paa_segments, merge_offset;
    bool lower_bounds;
    int rank = 0, ranks = 1, window = 0;
    int groups = 1, group = 0, group_size = 1;  // replication groups; this rank's group
    MPI_Comm group_comm = MPI_COMM_NULL;
    MPI_Win counter_window = MPI_WIN_NULL;     // next template of the group, on its first rank
    int *counter = nullptr;
    std::thread progress;                      // on the counter's rank: serves the other ranks' claims
    std::atomic<bool> stop_progress{false};
    idx_t first_subsequence = 0;       // global position of this rank's first subsequence
    idx_t subsequences = 0;            // subsequences owned by this rank
    std::vector<float> raw;            // raw samples of the owned subsequences
    std::vector<float> means, stds;    // per owned subsequence
    std::vector<sax_type> sax;         // paa_segments iSAX symbols per owned subsequence
    std::vector<sax_type> cardinalities;
    double summary_bytes = 0.0;
};

#endif
