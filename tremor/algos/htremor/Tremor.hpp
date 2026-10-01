#ifndef TREMOR_HPP
#define TREMOR_HPP

#include "../../distance_computers/DistanceComputer.hpp"
#include "bsf_sharing.hpp"
#include "workstealing.hpp"
#include "replication.hpp"
#include "../../utils/TimerManager.hpp"
#include "query_answering.hpp"

#include <queue>
#include <cfloat>
#include <string>
#include <stdexcept>
#include <omp.h>

#include <mpi.h>

struct TremorConfig
{
    int search_workers = 1;
    int index_threads = 1;
    int leaf_size = 2000;
    int paa_segments = 16;
    int replication_groups = 0;
    int query_threads = 1;
    int pq_th_div_factor = 16;
};

using ws_func_type = query_result (*)(WsSearchFunctionParams);

// How candidate windows are verified with exact distances.
//   Leaf:     in lower-bound order, leaf by leaf (classic TREMOR).
//   Adaptive: Leaf, unless more than the scan fallback fraction of a
//             sample of windows pass the lower bound; then a skip-sequential
//             scan in position order. Decided per query and per rank.
enum class RefinementMode
{
    Leaf,
    Adaptive
};

// Process-wide settings read by every query of the search.
void set_refinement_mode(RefinementMode mode);
void set_scan_fallback_fraction(double fraction);
// Adaptive refinement decisions of this rank so far.
void adaptive_refinement_counts(long long *leaf_queries, long long *scan_queries);

class Tremor
{

    friend void tremor_optimize_params(Tremor *tremor);
    friend void tremor_log_parameters(Tremor *tremor);
    friend void tremor_prepare_structures(Tremor *tremor);
    friend void print_index_stats(isax_index *index, int my_rank);
    friend void *index_creation_sequence_worker(void *transferdata);
    friend void tree_index_creation_from_pRecBuf_fai_blocking(void *transferdata);
    friend void tremor_preprocess_and_sort_queries(Tremor *tremor, TremorQuery *queries, int q_num, bool apply_sort);
    friend void tremor_perform_workstealing(Tremor *tremor, TremorQuery *queries, NodeList nodelist,
                                            ws_func_type ws_func, double (*estimation_func)(double),
                                            query_result *results, std::vector<BsfMessage> *shared_bsf_results);

private:
    float *database = nullptr;
    idx_t n_database = 0;
    idx_t dim = 0;
    DistanceType distance_type;
    int num_threads = 1;

    int search_workers = 1;
    int index_threads = 1;
    int replication_groups = 0;
    int query_threads = 1;
    int paa_segments = 16;
    int sax_cardinality = 8;
    int leaf_size = 2000;
    int initial_lbl_size = 2000;
    int initial_fbl_size = 100;

    idx_t dataset_size = 0;

    int min_leaf_size = 10;
    int time_series_size = 256;
    int density_aware_prepro = 0;
    std::string dataset_type = "default";
    int pq_th_div_factor = 16;
    int read_block_length = 20000;
    int flush_limit = 1000000;
    std::string output_file;
    bool bsf_sharing = false;
    std::string replication_groups_file;
    int workstealing_mode = 1;
    int ws_items_to_send = 0;
    std::string query_time_predictions_file;
    int mode = 1;
    int query_scheduling = 3;
    int merge_offset = 0;
    double index_memory_bytes = 0.0; // set by the index build
    float corr_threshold = 0.2f;
    bool verbose = false;
    int dynamic_scheduling_mode = 1;
    int top_k = 1;

    int my_rank = 0;
    int comm_sz = 1;

    float *rawfile = nullptr;

    int query_counter = 0;

    int ts_group_length = 1;

    idx_t long_sequence_length = 0;
    idx_t local_chunk_start = 0;
    idx_t local_chunk_length = 0;
    idx_t local_owned_subseq_start = 0;
    idx_t local_owned_subseq_count = 0;
    float *subsequence_means = nullptr;
    float *subsequence_stds = nullptr;

    BsfSharingData bsf_sharing_data;

    WorkstealingData workstealing_data;

    ReplicationData replication_data;

    isax_index_settings *index_settings = nullptr;
    isax_index *index = nullptr;

    query_result *results = nullptr;

    void initializeMPI(int argc, char **argv);

    float *buildIndexSequence(const float *database, const std::string *filename);

    void searchIndexL2Squared(TremorQuery *queries, int q_num, int topk,
                              query_result *results, std::vector<BsfMessage> *shared_bsf_results,
                              NodeList &nodelist, idx_t *I, float *D,
                              double (*basis_func)(double));

protected:
    bool validateSearchParams(const idx_t k, const idx_t n_query) const;

public:
    Tremor(DistanceType distance_type);

    Tremor(DistanceType distance_type, int argc, char **argv);

    Tremor(const TremorConfig &config, DistanceType distance_type, int argc, char **argv);
    void setNumThreads(int num_threads);
    int getNumThreads() const;
    void setModeKNN() { mode = 1; }
    void setModeThreshold() { mode = 0; }
    void setMergeOffset(int v) { merge_offset = v; }
    void setCorrelationThreshold(float v) { corr_threshold = v; }
    void setOutputFile(const std::string &path) { output_file = path; }
    void setVerbose(bool v) { verbose = v; }
    void setQueryThreads(int v) { query_threads = v; }
    void setIndexThreads(int v) { index_threads = v; }
    void setReplicationGroups(int v) { replication_groups = v; }
    void setRefinementMode(RefinementMode v) { set_refinement_mode(v); }

    int getMyRank() const { return my_rank; }
    double getIndexMemoryBytes() const { return index_memory_bytes; }
    int getCommSz() const { return comm_sz; }

    void buildIndex(float *database, idx_t n_database, idx_t dim);
    void buildIndexLongSequence(float *long_sequence, idx_t long_sequence_length, idx_t subsequence_length);
    void buildIndexLongSequence(const std::string &filename, idx_t long_sequence_length, idx_t subsequence_length);
    void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0);

    void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);

    int getResultCompareRank() const { return my_rank; }
    isax_index *getIndex() const { return index; }

    ~Tremor();
};

#include "../../isax/iSAXIndex.hpp"

#include "query_answering.hpp"

using index_func_type = ts_type *(*)(Tremor *);
using qa_func_type = query_result (*)(SearchFunctionParams);

using qa_sched_func_type = query_result *(*)(Tremor *, qa_func_type, ws_func_type);

TremorQuery *load_queries_from_buffer(const float *query_buf, int q_num, isax_index *index, int my_rank);
void free_queries(TremorQuery *queries, int q_num);
double predict_exec_time(float bsf, const char *dataset_type);
int cmp_query(const void *a, const void *b);

void send_initial_queries_module_coordinator_async_chatzakis(int *q_loaded, int my_rank, int comm_sz,
                                                             int distributed_queries_initial_burst,
                                                             int **process_buffer_initial,
                                                             int *rec_message, MPI_Request *request, MPI_Request *send_request,
                                                             int q_num, int *termination_message_id,
                                                             ReplicationData *replication_data);

int send_queries_module_coordinator_async_chatzakis(int *q_loaded, int q_num, int *process_buffer, MPI_Request *request, int *rec_message,
                                                    MPI_Request *send_request, int *termination_message_id,
                                                    ReplicationData *replication_data, int my_rank, int comm_sz,
                                                    bool verbose);

NodeList initialize_node_list(isax_index *index, int my_rank);

void tremor_optimize_params(Tremor *tremor);
void tremor_log_parameters(Tremor *tremor);
void tremor_prepare_structures(Tremor *tremor);

void tremor_preprocess_and_sort_queries(Tremor *tremor, TremorQuery *queries, int q_num, bool apply_sort);
void tremor_perform_workstealing(Tremor *tremor, TremorQuery *queries, NodeList nodelist,
                                 ws_func_type ws_func, double (*estimation_func)(double),
                                 query_result *results, std::vector<BsfMessage> *shared_bsf_results);

#endif
