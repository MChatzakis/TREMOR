#include "iSAXSearch.hpp"
#include "iSAXIndex.hpp"
#include "iSAXPqueue.hpp"

#include <float.h>
#include <stdlib.h>

    void calculate_node_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf, float *rawfile)
    {
        // TODO: implementation pending. Empty stub to satisfy the linker.
        (void)index;
        (void)node;
        (void)query;
        (void)pq_bsf;
        (void)rawfile;
    }

    void approximate_topk_inmemory(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float *rawfile)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);
        sax_from_paa(paa, sax, index->settings->paa_segments,
                     index->settings->sax_alphabet_cardinality,
                     index->settings->sax_bit_cardinality);

        root_mask_type root_mask = 0;
        CREATE_MASK(root_mask, index, sax);

        // index->fbl is a parallel_first_buffer_layer_ekosmas in Tremor (see
        // Tremor::buildIndexSequence). The two struct layouts are intentionally
        // compatible up to soft_buffers, but cast to the actual type for clarity
        // and to access the correct soft buffer element type.
        parallel_first_buffer_layer_ekosmas *p_fbl =
            reinterpret_cast<parallel_first_buffer_layer_ekosmas *>(index->fbl);

        if (p_fbl != nullptr &&
            (&p_fbl->soft_buffers[(int)root_mask])->initialized)
        {
            isax_node *node = (&p_fbl->soft_buffers[(int)root_mask])->node;

            // Walk down the tree using the SAX bits.
            while (node != nullptr && !node->is_leaf)
            {
                if (node->split_data == nullptr)
                    break;

                int location = index->settings->sax_bit_cardinality - 1 -
                               node->split_data->split_mask[node->split_data->splitpoint];
                root_mask_type mask = index->settings->bit_masks[location];

                if (sax[node->split_data->splitpoint] & mask)
                {
                    node = node->right_child;
                }
                else
                {
                    node = node->left_child;
                }
            }

            if (node != nullptr)
                calculate_node_topk_inmemory(index, node, ts, pq_bsf, rawfile);
        }

        free(sax);
    }

    void refine_topk_answer_inmemory(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float minimum_distance, int limit, float *rawfile)
    {
        (void)limit;

        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;

        int j = 0;
        pqueue_t *pq = pqueue_init(index->settings->root_nodes_size,
                                   cmp_pri, get_pri, set_pri, get_pos, set_pos);

        // Insert all root nodes in heap.
        isax_node *current_root_node = index->first_node;

        while (current_root_node != nullptr)
        {
            // Skip nodes with uninitialized isax data.
            if (current_root_node->isax_values == nullptr || current_root_node->isax_cardinalities == nullptr)
            {
                current_root_node = current_root_node->next;
                continue;
            }

            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));

            mindist_result->distance = minidist_paa_to_isax(paa, current_root_node->isax_values,
                                                            current_root_node->isax_cardinalities,
                                                            index->settings->sax_bit_cardinality,
                                                            index->settings->sax_alphabet_cardinality,
                                                            index->settings->paa_segments,
                                                            MINVAL, MAXVAL,
                                                            index->settings->mindist_sqrt);
            mindist_result->node = current_root_node;
            pqueue_insert(pq, mindist_result);

            current_root_node = current_root_node->next;
        }

        query_result *n;
        int checks = 0;
        while ((n = (query_result *)pqueue_pop(pq)))
        {
            if (!n->node)
            {
                free(n);
                continue;
            }
            // Skip nodes missing SAX metadata to avoid invalid accesses.
            if (n->node->isax_values == nullptr || n->node->isax_cardinalities == nullptr)
            {
                free(n);
                continue;
            }
            // The best node has a worse mindist, so search is finished.
            if (n->distance >= pq_bsf->knn[pq_bsf->k - 1] || n->distance > minimum_distance)
            {
                pqueue_insert(pq, n);
                break;
            }
            else
            {
                // If it is a leaf, check its real distance.
                if (n->node->is_leaf)
                {
                    // *** ADAPTIVE SPLITTING ***
                    // Only split if buffer exists (node hasn't been split already).
                    if (!n->node->has_full_data_file &&
                        (n->node->leaf_size > index->settings->min_leaf_size) &&
                        n->node->buffer != nullptr)
                    {
                        split_node(index, n->node);
                        // Only re-queue if split succeeded (node is no longer a leaf).
                        if (!n->node->is_leaf)
                        {
                            pqueue_insert(pq, n);
                            continue;
                        }
                        // If still a leaf (split failed), fall through to process normally.
                    }
                    // *** EXTRA BOUNDING ***
                    if (tight_bound)
                    {
                        j++;
                        float mindistance = calculate_minimum_distance_inmemory(index, n->node, ts, paa);

                        if (mindistance >= pq_bsf->knn[pq_bsf->k - 1])
                        {
                            free(n);
                            continue;
                        }
                    }
                    // *** REAL DISTANCE ***
                    checks++;
                    calculate_node_topk_inmemory(index, n->node, ts, pq_bsf, rawfile);

                    if (pq_bsf->knn[pq_bsf->k - 1] < FLT_MAX)
                    {
                        pqueue_insert(pq, n);
                        break;
                    }
                }
                else
                {
                    // Intermediate node: push children with their mindist.
                    if (n->node->left_child != nullptr &&
                        n->node->left_child->isax_values != nullptr &&
                        n->node->left_child->isax_cardinalities != nullptr)
                    {
                        if (n->node->left_child->is_leaf && !n->node->left_child->has_partial_data_file && aggressive_check)
                        {
                            calculate_node_topk_inmemory(index, n->node->left_child, ts, pq_bsf, rawfile);
                        }
                        else
                        {
                            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
                            mindist_result->distance = minidist_paa_to_isax(paa, n->node->left_child->isax_values,
                                                                            n->node->left_child->isax_cardinalities,
                                                                            index->settings->sax_bit_cardinality,
                                                                            index->settings->sax_alphabet_cardinality,
                                                                            index->settings->paa_segments,
                                                                            MINVAL, MAXVAL,
                                                                            index->settings->mindist_sqrt);
                            mindist_result->node = n->node->left_child;
                            pqueue_insert(pq, mindist_result);
                        }
                    }
                    if (n->node->right_child != nullptr &&
                        n->node->right_child->isax_values != nullptr &&
                        n->node->right_child->isax_cardinalities != nullptr)
                    {
                        if (n->node->right_child->is_leaf && !n->node->right_child->has_partial_data_file && aggressive_check)
                        {
                            calculate_node_topk_inmemory(index, n->node->right_child, ts, pq_bsf, rawfile);
                        }
                        else
                        {
                            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
                            mindist_result->distance = minidist_paa_to_isax(paa, n->node->right_child->isax_values,
                                                                            n->node->right_child->isax_cardinalities,
                                                                            index->settings->sax_bit_cardinality,
                                                                            index->settings->sax_alphabet_cardinality,
                                                                            index->settings->paa_segments,
                                                                            MINVAL, MAXVAL,
                                                                            index->settings->mindist_sqrt);
                            mindist_result->node = n->node->right_child;
                            pqueue_insert(pq, mindist_result);
                        }
                    }
                }

                // Free the node currently popped.
                free(n);
            }
        }

        // Free the nodes that were not popped.
        while ((n = (query_result *)pqueue_pop(pq)))
        {
            free(n);
        }

        // NOTE: do NOT overwrite pq_bsf->knn[i] with pq_bsf->knn[k-1] here.
        // The previous version of this function (and the original
        // calculate_node_topk_inmemory) ended with such a loop, which destroyed
        // the sorted top-k results by replacing every entry with the worst one.
        // The BSF queue is already correctly maintained by pqueue_bsf_insert
        // inside calculate_node_topk_inmemory.

        // Free the priority queue.
        pqueue_free(pq);
    }
