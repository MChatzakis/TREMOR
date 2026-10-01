#ifndef QUERY_ANSWERING_HPP
#define QUERY_ANSWERING_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include <cstdbool>
#include <sched.h>
#include <utility>

#include <mpi.h>

#include "../../isax/iSAXTypes.hpp"
#include "../../isax/iSAXIndex.hpp"
#include "../../isax/iSAXPqueue.hpp"
#include "../../utils/TimerManager.hpp"
#include "workstealing.hpp"
#include "replication.hpp"
#include "bsf_sharing.hpp"
#include "indexing.hpp"

class Tremor;

inline int cmp_pri(double next, double curr)
{
    return (next > curr);
}

inline double get_pri(void *a)
{
    return (double)((query_result *)a)->distance;
}

inline void set_pri(void *a, double pri)
{
    ((query_result *)a)->distance = (float)pri;
}

inline size_t get_pos(void *a)
{
    return ((query_result *)a)->pqueue_position;
}

inline void set_pos(void *a, size_t pos)
{
    ((query_result *)a)->pqueue_position = pos;
}

struct TremorQuery
{
    int id;
    ts_type *query;
    ts_type *paa;
    double initial_estimation;
    pqueue_bsf *initial_pq_bsfs;
};

struct SearchFunctionParams
{
    int query_id;
    ts_type *ts;
    ts_type *paa;
    isax_index *index;
    NodeList *nodelist;
    float minimum_distance;

    double (*estimation_func)(double);
    CommunicationModuleData *comm_data;

    std::vector<BsfMessage> *shared_bsf_results;
    pqueue_bsf *precomputed_bsfs;

    int k;
    int my_rank;
    int comm_sz;
    int query_threads;
    bool verbose;
    float *rawfile;
    int merge_offset;
    int query_counter;
    int pq_th_div_factor;
    float corr_threshold;

    BsfSharingData *bsf_sharing_data;
    WorkstealingData *workstealing_data;
    ReplicationData *replication_data;

    std::string output_file;

    int warp_window;
    ts_type *paaU;
    ts_type *paaL;

    idx_t local_chunk_start;
    idx_t local_chunk_length;
    idx_t local_owned_subseq_start;
    idx_t local_owned_subseq_count;
    float *subsequence_means;
    float *subsequence_stds;
    int subsequence_length;

    bool threshold_mode;
    float threshold_ed;
    FILE *threshold_output_file;
    pthread_mutex_t *threshold_output_lock;
    idx_t *threshold_match_counter;
    std::vector<std::pair<file_position_type, float>> *threshold_hits;
    pthread_mutex_t *threshold_hits_lock;
};

struct WsSearchFunctionParams
{
    int query_id;
    ts_type *ts;
    ts_type *paa;
    isax_index *index;
    isax_node *lca_node;
    NodeList *nodelist;
    float minimum_distance;
    float bsf;
    file_position_type bsf_pos;
    int *batch_ids;
    double (*estimation_func)(double);

    CommunicationModuleData *comm_data;

    std::vector<BsfMessage> *shared_bsf_results;
    pqueue_bsf *precomputed_bsfs;

    int k;
    int my_rank;
    int comm_sz;
    int query_threads;
    bool verbose;
    float *rawfile;
    int merge_offset;
    int query_counter;
    int pq_th_div_factor;
    float corr_threshold;

    BsfSharingData *bsf_sharing_data;
    WorkstealingData *workstealing_data;
    ReplicationData *replication_data;

    std::string output_file;

    int warp_window;
    ts_type *paaU;
    ts_type *paaL;

    idx_t local_chunk_start;
    idx_t local_chunk_length;
    idx_t local_owned_subseq_start;
    idx_t local_owned_subseq_count;
    float *subsequence_means;
    float *subsequence_stds;
    int subsequence_length;

    bool threshold_mode;
    float threshold_ed;
    FILE *threshold_output_file;
    pthread_mutex_t *threshold_output_lock;
    idx_t *threshold_match_counter;
    std::vector<std::pair<file_position_type, float>> *threshold_hits;
    pthread_mutex_t *threshold_hits_lock;
};

struct QaWorkerData
{
    int workernumber;
    float minimum_distance;

    ts_type *paa, *ts;
    // Block visiting order for the early-abandoning distance (see
    // query_block_order in Tremor.cpp).
    const int *block_order;
    // Cursor over owned windows for the position-ordered scan.
    // kNN with a merge offset: every verified window within the pruning
    // bound (position, squared distance), else nullptr. Guarded by bsf_lock.
    std::vector<std::pair<file_position_type, float>> *knn_candidates;
    volatile long long *refine_cursor;
    // RefinementMode::Adaptive: shared decision (1 = scan), else nullptr,
    // and the shared count of sampled candidate windows for the pre-check.
    volatile int *adaptive_decision;
    volatile long long *adaptive_sample_candidates;
    volatile long long *adaptive_sample_blocks; // scan cost estimate for the FFT fallback (fft_fallback.hpp)
    // kNN: the leaf of the query's own iSAX word, verified first (or nullptr).
    isax_node *seed_leaf;
    bool share_seed; // kNN without replication: share the first bounds of all ranks
    bool record_lb;  // compute the lower bounds of single windows (they prune)
    isax_index *index;

    query_result *bsf_result;

    SubtreeBatch *batches;
    int total_batches;

    volatile int *batch_counter;
    volatile int *pq_counter;

    pthread_barrier_t *sync_barrier;
    pthread_mutex_t *bsf_lock;

    volatile char *receiving_workstealing;
    volatile char *priority_queues_filled;

    pqueue_t ***final_pq_list;
    int *final_pq_list_size;

    pthread_mutex_t *distances_lock;

    int *pqs_stolen;
    int *processed_pqs;

    CommunicationModuleData *comm_data;

    std::vector<BsfMessage> *shared_bsf_results;

    int my_rank;
    int comm_sz;
    float *rawfile;
    int query_threads;
    int merge_offset;
    int query_counter;
    int pq_th_div_factor;

    float corr_threshold;

    bool verbose;

    ReplicationData *replication_data;
    WorkstealingData *workstealing_data;
    BsfSharingData *bsf_sharing_data;

    std::string output_file;

    int warp_window;
    ts_type *paaU;
    ts_type *paaL;

    idx_t local_chunk_start;
    idx_t local_chunk_length;
    idx_t local_owned_subseq_start;
    idx_t local_owned_subseq_count;
    float *subsequence_means;
    float *subsequence_stds;
    int subsequence_length;

    bool threshold_mode;
    float threshold_ed;
    FILE *threshold_output_file;
    pthread_mutex_t *threshold_output_lock;
    idx_t *threshold_match_counter;
    std::vector<std::pair<file_position_type, float>> *threshold_hits;
    pthread_mutex_t *threshold_hits_lock;
};

struct CoordinatorData
{
    CommunicationModuleData *comm_data;
    volatile char *threads_finished;
};

struct WorkstealingThreadData
{
    volatile char *query_workers_finished;
    volatile char *priority_queues_filled;
    volatile char *receiving_workstealing;

    pthread_mutex_t *bsf_lock;
    query_result *bsf_result;

    BatchList *batchlist;

    int *pqs_stolen;
    int *workstealing_times;

    pqueue_t ***final_pq_list;
    int *final_pq_list_size;

    isax_index *index;

    int my_rank;
    int comm_sz;
    float *rawfile;
    int query_threads;
    int merge_offset;
    int query_counter;
    int pq_th_div_factor;

    float corr_threshold;

    bool verbose;

    ReplicationData *replication_data;
    WorkstealingData *workstealing_data;
    BsfSharingData *bsf_sharing_data;

    std::string output_file;
};

float calculate_minimum_distance_inmemory(isax_index *index, isax_node *node, ts_type *raw_query, ts_type *query);
int process_pq_of_batch_chatzakis(int current_pq_index, QaWorkerData *input_data);
void gather_sort_pqueues(QaWorkerData *in_data);
int estimate_th(double x, double (*estimation_func)(double));

void process_rs_batch(int batch_index, SubtreeBatch *batches, float bsf_distance, isax_index *index, ts_type *paa);
void generate_pqs_of_rs_batch(isax_node *subtree_node, SubtreeBatch *batch, float bsf_distance, ts_type *paa, isax_index *index);

void *workstealing_manager(void *rfdata);
void *dynamic_query_scheduler(void *rfdata);
void *qa_exact_search_messi_worker(void *rfdata);
void *exact_search_worker_inmemory_hybridpqueue_parallel_chatzakis(void *rfdata);
void *qa_exact_search_messi_dynamic_worker(void *rfdata);
void *qa_exact_search_tremor_worker(void *rfdata);

query_result qa_exact_search_tremor_knn(SearchFunctionParams args);

query_result qa_exact_search_tremor_knn_workstealing(WsSearchFunctionParams ws_args);

#endif
