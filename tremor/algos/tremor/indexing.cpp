#include "../htremor/indexing.hpp"
#include "../../isax/SAX.hpp"
#include "../../utils/TimerManager.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <climits>
#include <algorithm>
#include <mutex>

constexpr float TREMOR_MISSING_VALUE = -100000.0f;

#ifndef CHECK_ALLOC
#define CHECK_ALLOC(ptr, rank)                                                                               \
    if ((ptr) == nullptr)                                                                                    \
    {                                                                                                        \
        fprintf(stderr, "[Node %d] Error: Memory allocation failed in %s:%d\n", (rank), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE);                                                                             \
    }
#endif

void *index_creation_sequence_worker(void *transferdata)
{
    buffer_data_inmemory_ekosmas *input_data =
        static_cast<buffer_data_inmemory_ekosmas *>(transferdata);

    float *rawfile = input_data->rawfile;
    const idx_t local_owned_subseq_count = input_data->local_owned_subseq_count;
    const idx_t local_owned_subseq_start = input_data->local_owned_subseq_start;
    const idx_t local_chunk_start = input_data->local_chunk_start;
    const idx_t local_chunk_length = input_data->local_chunk_length;
    const int subsequence_length = input_data->subsequence_length;

    if (local_owned_subseq_count == 0 || subsequence_length <= 0)
    {
        pthread_barrier_wait(input_data->wait_summaries_to_compute);
        tree_index_creation_from_pRecBuf_fai_blocking(transferdata);
        return nullptr;
    }

    unsigned long ts_num = static_cast<unsigned long>(input_data->ts_num);
    unsigned long total_blocks = (ts_num == 0)
                                     ? 0
                                     : ((ts_num - 1UL) / static_cast<unsigned long>(input_data->readblock));

    isax_index *index = input_data->index;
    unsigned long localcounterblock =
        static_cast<unsigned long>(input_data->workernumber);
    unsigned long *next_block_to_process = input_data->shared_start_number;

    if (input_data->deterministic_index)
    {

        next_block_to_process = &localcounterblock;
    }

    int paa_segments = index->settings->paa_segments;
    int sax_byte_size = index->settings->sax_byte_size;

    file_position_type pos;
    sax_type *sax = static_cast<sax_type *>(std::malloc(static_cast<size_t>(sax_byte_size)));
    CHECK_ALLOC(sax, input_data->my_rank);

    ts_type *normalized_subsequence = static_cast<ts_type *>(
        std::malloc(sizeof(ts_type) * static_cast<size_t>(subsequence_length)));
    CHECK_ALLOC(normalized_subsequence, input_data->my_rank);

    if (input_data->workernumber == 0)
    {
    }

    unsigned long i, block_num, my_ts_start, my_ts_end;
    while (true)
    {
        if (input_data->deterministic_index)
        {
            block_num = __sync_fetch_and_add(next_block_to_process,
                                             static_cast<unsigned long>(input_data->index_threads));
        }
        else
        {
            block_num = __sync_fetch_and_add(next_block_to_process, 1UL);
        }

        if (block_num > total_blocks)
        {
            break;
        }

        my_ts_start = block_num * static_cast<unsigned long>(input_data->readblock);
        if (block_num == total_blocks)
        {
            my_ts_end = ts_num;
        }
        else
        {
            my_ts_end =
                (block_num + 1UL) * static_cast<unsigned long>(input_data->readblock);
        }

        if (my_ts_start >= my_ts_end)
        {
            continue;
        }

        idx_t local_subseq_id = static_cast<idx_t>(my_ts_start);
        idx_t global_start = local_owned_subseq_start + local_subseq_id;
        idx_t local_raw_start = global_start - local_chunk_start;

        if (local_raw_start + static_cast<idx_t>(subsequence_length) > local_chunk_length)
        {
            continue;
        }

        for (i = my_ts_start; i < my_ts_end; i++)
        {
            local_subseq_id = static_cast<idx_t>(i);
            global_start = local_owned_subseq_start + local_subseq_id;
            local_raw_start = global_start - local_chunk_start;

            if (local_raw_start + static_cast<idx_t>(subsequence_length) > local_chunk_length)
            {
                break;
            }

            if (rawfile[local_raw_start] != TREMOR_MISSING_VALUE)
            {
                // Exact two-pass statistics. A running sum/sum_sq loses every
                // significant digit in near-silent windows (gaps, taper edges)
                // that follow large-amplitude samples.
                const float *window = rawfile + local_raw_start;
                double sum = 0.0;
                for (int d = 0; d < subsequence_length; d++)
                {
                    if (window[d] != TREMOR_MISSING_VALUE)
                        sum += static_cast<double>(window[d]);
                }
                const double mean = sum / static_cast<double>(subsequence_length);
                double sum_sq_dev = 0.0;
                for (int d = 0; d < subsequence_length; d++)
                {
                    const double v = (window[d] == TREMOR_MISSING_VALUE) ? 0.0 : static_cast<double>(window[d]);
                    sum_sq_dev += (v - mean) * (v - mean);
                }
                const double stddev = std::sqrt(sum_sq_dev / static_cast<double>(subsequence_length));
                const bool degenerate = stddev <= 1e-12;
                const double denom = degenerate ? 1.0 : stddev;

                input_data->subsequence_means[local_subseq_id] = static_cast<float>(mean);
                input_data->subsequence_stds[local_subseq_id] = static_cast<float>(degenerate ? 0.0 : stddev);

                for (int d = 0; d < subsequence_length; d++)
                {
                    const float raw_sample = rawfile[local_raw_start + static_cast<idx_t>(d)];
                    const double raw_v = (raw_sample == TREMOR_MISSING_VALUE)
                                             ? 0.0
                                             : static_cast<double>(raw_sample);
                    normalized_subsequence[d] = static_cast<ts_type>((raw_v - mean) / denom);
                }

                if (sax_from_ts(normalized_subsequence,
                                sax,
                                index->settings->ts_values_per_paa_segment,
                                paa_segments,
                                index->settings->sax_alphabet_cardinality,
                                index->settings->sax_bit_cardinality) == SUCCESS)
                {
                    std::memcpy(&input_data->subsequence_sax[local_subseq_id * paa_segments], sax,
                                sizeof(sax_type) * paa_segments);
                    pos = static_cast<file_position_type>(global_start);

                    isax_pRecBuf_index_insert_inmemory_ekosmas(
                        index,
                        sax,
                        &pos,
                        input_data->lock_firstnode,
                        input_data->workernumber,
                        input_data->index_threads);
                }
                else
                {
                    fprintf(stderr,
                            "error: cannot insert record in index, since sax representation "
                            "failed to be created\n");
                    std::exit(EXIT_FAILURE);
                }
            }
            else
            {
                input_data->subsequence_means[local_subseq_id] = 0.0f;
                input_data->subsequence_stds[local_subseq_id] = 0.0f;
            }
        }
    }

    std::free(sax);
    std::free(normalized_subsequence);

    pthread_barrier_wait(input_data->wait_summaries_to_compute);

    tree_index_creation_from_pRecBuf_fai_blocking(transferdata);

    return nullptr;
}

void tree_index_creation_from_pRecBuf_fai_blocking(void *transferdata)
{
    buffer_data_inmemory_ekosmas *input_data =
        static_cast<buffer_data_inmemory_ekosmas *>(transferdata);
    isax_index *index = input_data->index;
    int j;
    isax_node_record *r = static_cast<isax_node_record *>(std::malloc(sizeof(isax_node_record)));
    if (r == nullptr)
    {
        fprintf(stderr, "[Node %d] Error: Memory allocation failed for isax_node_record\n",
                input_data->my_rank);
        std::exit(EXIT_FAILURE);
    }

    idx_t worker_inserts = 0;

    while (true)
    {

        j = __sync_fetch_and_add(input_data->node_counter, 1);
        if (j >= index->fbl->number_of_buffers)
        {
            break;
        }

        parallel_first_buffer_layer_ekosmas *p_fbl =
            reinterpret_cast<parallel_first_buffer_layer_ekosmas *>(index->fbl);
        parallel_fbl_soft_buffer_ekosmas *current_fbl_node = &(p_fbl->soft_buffers[j]);

        if (!current_fbl_node->initialized)
        {
            continue;
        }

        for (int k = 0; k < input_data->index_threads; k++)
        {
            for (int i = 0; i < current_fbl_node->buffer_size[k]; i++)
            {

                r->sax = static_cast<sax_type *>(
                    &(current_fbl_node->sax_records[k][i * index->settings->paa_segments]));
                r->position = static_cast<file_position_type *>(
                    &(static_cast<file_position_type *>(current_fbl_node->pos_records[k])[i]));
                r->insertion_mode = static_cast<insertion_mode>(NO_TMP | PARTIAL);

                worker_inserts++;

                if (current_fbl_node->node == nullptr)
                {

                    continue;
                }

                isax_node *result_node = add_record_to_node_inmemory(
                    index,
                    static_cast<isax_node *>(current_fbl_node->node),
                    r,
                    1);

                if (result_node == nullptr)
                {
                    continue;
                }
            }
        }
    }

    printf("[Node %d] Worker %d inserted %llu records\n",
           input_data->my_rank,
           input_data->workernumber,
           static_cast<unsigned long long>(worker_inserts));

    std::free(r);
}

long int find_total_nodes(isax_node *root_node)
{
    long int c = 1;
    if (root_node == nullptr)
    {
        return 0;
    }
    else
    {
        c += find_total_nodes(root_node->left_child);
        c += find_total_nodes(root_node->right_child);
        return c;
    }
}

long int find_total_leafs_nodes(isax_node *root_node)
{
    if (root_node == nullptr)
    {
        return 0;
    }

    if (root_node->left_child == nullptr && root_node->right_child == nullptr)
    {
        return 1;
    }
    else
    {
        return find_total_leafs_nodes(root_node->left_child) +
               find_total_leafs_nodes(root_node->right_child);
    }
}

long int find_tree_height(isax_node *root_node)
{
    if (root_node == nullptr)
    {
        return -1;
    }
    else
    {
        long int left_depth = find_tree_height(root_node->left_child);
        long int right_depth = find_tree_height(root_node->right_child);

        if (left_depth > right_depth)
        {
            return left_depth + 1;
        }
        else
        {
            return right_depth + 1;
        }
    }
}

long int count_ts_in_nodes(isax_node *root_node, const char parallelism_in_subtree,
                           const char recBuf_helpers_exist)
{
    long int my_subtree_nodes = 0;

    if (!root_node->is_leaf)
    {
        my_subtree_nodes = count_ts_in_nodes(root_node->left_child, parallelism_in_subtree, recBuf_helpers_exist);
        my_subtree_nodes += count_ts_in_nodes(root_node->right_child, parallelism_in_subtree, recBuf_helpers_exist);
        return my_subtree_nodes;
    }
    else
    {

        return root_node->leaf_size;
    }
}

long int find_total_nodes_tmp(isax_node *root_node)
{
    long int c = 1;
    if (root_node == nullptr)
    {
        return 0;
    }
    else
    {
        c += find_total_nodes(root_node->left_child);
        c += find_total_nodes(root_node->right_child);
        return c;
    }
}

long int find_tree_height_tmp(isax_node *root_node)
{
    if (root_node == nullptr)
    {
        return -1;
    }
    else
    {
        long int left_depth = find_tree_height(root_node->left_child);
        long int right_depth = find_tree_height(root_node->right_child);

        if (left_depth > right_depth)
        {
            return left_depth + 1;
        }
        else
        {
            return right_depth + 1;
        }
    }
}

void print_index_stats(isax_index *index, int my_rank)
{
    long int empty_subtrees_buffers = 0;
    int non_empty_subtrees_cnt = 0;

    long int total_nodes = 0;
    long int tree_height = 0;

    long int total_leafs_nodes = 0;

    parallel_first_buffer_layer_ekosmas *fbl_ekosmas =
        reinterpret_cast<parallel_first_buffer_layer_ekosmas *>(index->fbl);

    for (int i = 0; i < fbl_ekosmas->number_of_buffers; i++)
    {
        parallel_fbl_soft_buffer_ekosmas *current_fbl_node = &fbl_ekosmas->soft_buffers[i];
        if (!current_fbl_node->initialized)
        {
            empty_subtrees_buffers++;
            continue;
        }

        non_empty_subtrees_cnt++;

        isax_node *subtree_root = current_fbl_node->node;
        long subtree_nodes = find_total_nodes(subtree_root);
        long subtree_height = find_tree_height(subtree_root);
        long subtree_leafs = find_total_leafs_nodes(subtree_root);

        total_nodes += subtree_nodes;
        tree_height += subtree_height;
        total_leafs_nodes += subtree_leafs;

        if (subtree_nodes > 1)
        {
            printf("Subtree[%d]: (height=%lu, nodes=%lu, leafs=%lu)\n", i, subtree_height, subtree_nodes, subtree_leafs);
        }
    }

    printf("\n--------------------\n[Node %d]: Tree Stats:\n"
           "Total buffers(2^16): %d\n"
           "Total tree nodes: %ld\n"
           "Total leafs nodes: %ld\n"
           "Average subtree height: %f\n"

           "Empty subtrees: %ld\n"
           "Non Empty subtrees: %d\n-------------------\n",
           my_rank,
           fbl_ekosmas->number_of_buffers,
           total_nodes,
           total_leafs_nodes,
           non_empty_subtrees_cnt > 0 ? (float)tree_height / non_empty_subtrees_cnt : 0.0f,

           empty_subtrees_buffers,
           non_empty_subtrees_cnt);
}

// Each batch holds MAX_PQs_WORKSTEALING + 1 queue slots (512 KB), so a
// fresh batch array costs a query milliseconds of page zeroing. The array
// is kept for the next query, which clears only the slots used before.
static std::mutex batch_cache_lock;
static SubtreeBatch *batch_cache = nullptr;
static int batch_cache_size = 0;
static bool batch_cache_used = false;

static SubtreeBatch *acquire_batches(int count)
{
    {
        std::lock_guard<std::mutex> guard(batch_cache_lock);
        if (batch_cache != nullptr && !batch_cache_used && batch_cache_size >= count)
        {
            batch_cache_used = true;
            for (int i = 0; i < batch_cache_size; i++)
            {
                const int used = std::min(std::max(batch_cache[i].pq_amount, 0), MAX_PQs_WORKSTEALING);
                std::fill(batch_cache[i].pq, batch_cache[i].pq + used + 1, nullptr);
            }
            return batch_cache;
        }
    }
    SubtreeBatch *batches = static_cast<SubtreeBatch *>(std::calloc(static_cast<size_t>(count), sizeof(SubtreeBatch)));
    if (batches == nullptr)
    {
        fprintf(stderr, "Error: Memory allocation failed for SubtreeBatch array\n");
        std::exit(EXIT_FAILURE);
    }
    return batches;
}

void release_subtree_batches(BatchList *batchlist)
{
    std::lock_guard<std::mutex> guard(batch_cache_lock);
    if (batchlist->batches == batch_cache)
        batch_cache_used = false;
    else if (batch_cache == nullptr && batchlist->batches != nullptr)
    {
        batch_cache = batchlist->batches;
        batch_cache_size = batchlist->batch_amount;
    }
    else
        std::free(batchlist->batches);
    std::free(batchlist);
}

BatchList *create_subtree_batches(NodeList *nodelist, int number_of_batches_to_create, int pq_th)
{
    if (nodelist == nullptr || nodelist->node_amount <= 0 || number_of_batches_to_create <= 0)
    {
        BatchList *empty_batchlist = static_cast<BatchList *>(std::malloc(sizeof(BatchList)));
        if (empty_batchlist == nullptr)
        {
            fprintf(stderr, "Error: Memory allocation failed for empty BatchList\n");
            std::exit(EXIT_FAILURE);
        }
        empty_batchlist->batches = nullptr;
        empty_batchlist->batch_amount = 0;
        return empty_batchlist;
    }

    if (nodelist->node_amount < number_of_batches_to_create)
    {
        number_of_batches_to_create = nodelist->node_amount;
    }

    BatchList *batchlist = static_cast<BatchList *>(std::malloc(sizeof(BatchList)));
    if (batchlist == nullptr)
    {
        fprintf(stderr, "Error: Memory allocation failed for BatchList\n");
        std::exit(EXIT_FAILURE);
    }

    batchlist->batch_amount = number_of_batches_to_create;
    batchlist->batches = acquire_batches(number_of_batches_to_create);

    int batch_size = nodelist->node_amount / batchlist->batch_amount;
    for (int i = 0; i < number_of_batches_to_create; i++)
    {
        SubtreeBatch &batch = batchlist->batches[i];
        batch.id = i;
        batch.from = i * batch_size;
        batch.to = i == number_of_batches_to_create - 1 ? nodelist->node_amount : (i + 1) * batch_size;
        batch.size = batch.to - batch.from;
        batch.current_subtree_to_process = 0;
        batch.processed_phase_1 = 0;
        batch.is_getting_help_phase1 = 0;
        batch.pq_th = pq_th;
        batch.pq_amount = 0;
        pthread_mutex_init(&batch.pq_insert_lock, nullptr);
        batch.nodelist = nodelist;
        batch.max_pq_index = 0;
        batch.min_pq_index = INT32_MAX;
        batch.is_stolen = 0;
    }

    return batchlist;
}
