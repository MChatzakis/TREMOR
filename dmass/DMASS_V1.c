#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <errno.h>
#include <fftw3.h>
#include <mpi.h>
#include <string.h>

#include <sys/time.h>

typedef unsigned long long file_position_type;
typedef unsigned long long data_size_type;

#define MASTER 0
#define MISSING_VALUE -100000
#define ENABLE_PER_QUERY_PRINT true

typedef enum
{
    MASS_KNN = 0,
    MASS_THRESHOLD = 1
} MASS_MODE;

typedef struct system_node_t
{
    int node_id;
    int threads;
} system_node_t;
typedef struct rep_group_t
{
    int group_id;
    int coordinator_node;
    int total_nodes;
    system_node_t *nodes;
    data_size_type total_time_series;
} rep_group_t;
typedef struct rep_data_t
{
    int total_groups;
    char *node_groups_config;
    int *node_group_mappings;
    rep_group_t *node_groups;
} rep_data_t;
void rep_init(data_size_type dataset_size, int num_threads);
void rep_destroy();
data_size_type rep_allocate_data_series_default(int group_id, data_size_type dataset_size);
int rep_find_group(int rank);
int rep_find_coordinator_node_rank(int rank);
int rep_get_repgroup_nodes(int rank);
bool rep_is_last_node_of_group(int rank);
data_size_type rep_get_time_series_of_group(int rank);
data_size_type rep_get_time_series_offset(int rank);
rep_group_t rep_get_group(int rank);

typedef struct mass_ctx_t
{
    double *x;          // local chunk (not owned)
    double *mu;         // per-window mean, n - m + 1 entries
    double *sigma;      // per-window population std, n - m + 1 entries
    double fft_dot_min; // below this m * sigma, the FFT dot product is recomputed directly
    int n, m, fft_len, num_threads;

    double *buf;     // fft_len reals: reversed query in, correlation out
    fftw_complex *X; // cached spectrum of x
    fftw_complex *Y; // per-query spectrum
    fftw_plan query_forward, query_backward;
} mass_ctx_t;
void mass_ctx_init(mass_ctx_t *ctx, double *x, int n, int m, int num_threads);
void mass_ctx_query(mass_ctx_t *ctx, double *y, double *dist);
void mass_ctx_destroy(mass_ctx_t *ctx);
double *zNorm(double *x, int n, double *y);

typedef struct pqueue_bsf
{
    int k;
    file_position_type *position;
    float *knn;
} pqueue_bsf;
pqueue_bsf *pqueue_bsf_init(int k);
void pqueue_bsf_insert_offset(pqueue_bsf *q, float data, file_position_type position, int pos_offset);
void pqueue_bsf_insert(pqueue_bsf *q, float data, file_position_type position);
void pqueue_bsf_destroy(pqueue_bsf *q);
double *load_data(char *ifilename, data_size_type total_samples, data_size_type *loaded_samples, int window_size);
void collect_knn_results(pqueue_bsf **knn_pqueues, data_size_type queries_size, char *output, int merge_offset, int window_size, int k);

/*
 * kNN with a merge offset (exclusion zone). The answer is canonical: sort all
 * windows by (distance, position) and keep a window unless it lies within
 * merge_offset of one already kept, until k are kept. The search keeps 2k
 * windows pairwise merge_offset apart (pqueue_bsf_insert_offset); each kept
 * answer window excludes at most 2 of them, so the 2k-th distance bounds the
 * k-th answer. Every window within that bound is a candidate, and the master
 * selects the answer from the candidates of all ranks. TREMOR implements the
 * same rule.
 */
typedef struct knn_candidate_t
{
    file_position_type position;
    float distance;
} knn_candidate_t;

typedef struct knn_candidates_t
{
    knn_candidate_t *items;
    size_t count, capacity;
} knn_candidates_t;

knn_candidates_t *KNN_CANDIDATES = NULL; // one list per query, kNN with a merge offset only

void knn_candidates_add(knn_candidates_t *list, file_position_type position, float distance);
int select_knn_with_exclusion(knn_candidate_t *candidates, size_t count, int k, int merge_offset, knn_candidate_t *answer);
void collect_qa_times(double qa_time);
pqueue_bsf **static_scheduling(const char *ifilename, data_size_type q_num, int window_size,
                               mass_ctx_t *mass, MASS_MODE mode, int k, float threshold,
                               int merge_offset, FILE *my_output_file);
void collect_and_write_timings(double index_time, double query_time, const char *path);

rep_data_t REPLICATION_CONF;
int MY_RANK, COMM_SZ;
bool VERBOSE = false;

int main(int argc, char *argv[])
{
    struct timeval current_time, total_time_start, qa_time_start, collect_time_start;
    double total_time, qa_time, collect_time;
    gettimeofday(&total_time_start, NULL);

    // 1. Initialize params and env
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &COMM_SZ);
    MPI_Comm_rank(MPI_COMM_WORLD, &MY_RANK);

    REPLICATION_CONF.total_groups = 0;

    // 2. Read the parameters
    char *dataset = NULL, *queries = NULL, *output = NULL;
    char *benchmark_output = NULL;
    bool skip_benchmark_report = false;
    data_size_type dataset_size = 0, queries_size = 0;
    MASS_MODE mode = MASS_THRESHOLD;
    int window_length = 0, k = 10, num_threads = 128, merge_offset = 0;
    float corr_th = 0.5;
    while (1)
    {
        static struct option long_options[] = {
            {"dataset", required_argument, 0, '0'},
            {"dataset-size", required_argument, 0, '1'},
            {"queries", required_argument, 0, '2'},
            {"queries-size", required_argument, 0, '3'},
            {"output", required_argument, 0, '4'},
            {"threads", required_argument, 0, '5'},
            {"mode", required_argument, 0, '6'},
            {"threshold", required_argument, 0, '7'},
            {"k", required_argument, 0, '8'},
            {"replication-groups", required_argument, 0, '9'},
            {"window-length", required_argument, 0, 'a'},
            {"merge-offset", required_argument, 0, 'b'},
            {"verbose", no_argument, 0, 'c'},
            {"help", no_argument, 0, 'd'},
            {"benchmark-output", required_argument, 0, 'e'},
            {"no-benchmark-report", no_argument, 0, 'f'},
            {NULL, 0, NULL, 0}};

        int option_index = 0;
        int c = getopt_long(argc, argv, "", long_options, &option_index);
        if (c == -1)
            break;
        switch (c)
        {
        case '0':
            dataset = optarg;
            break;
        case '1':
            sscanf(optarg, "%llu", &dataset_size);
            break;
        case '2':
            queries = optarg;
            break;
        case '3':
            sscanf(optarg, "%llu", &queries_size);
            break;
        case '4':
            output = optarg;
            break;
        case '5':
            num_threads = atoi(optarg);
            break;
        case '6':
            if (strcmp(optarg, "knn-search") == 0)
            {
                mode = MASS_KNN;
            }
            else if (strcmp(optarg, "threshold-search") == 0)
            {
                mode = MASS_THRESHOLD;
            }
            else
            {
                printf("Unknown mode: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case '7':
            corr_th = atof(optarg);
            break;
        case '8':
            k = atoi(optarg);
            break;
        case '9':
        {
            char *endptr = NULL;
            long requested_groups = strtol(optarg, &endptr, 10);
            if (endptr == optarg || *endptr != '\0' || requested_groups < 0 || requested_groups > INT_MAX)
            {
                if (MY_RANK == MASTER)
                    printf("[Node %d]: Error: --replication-groups must be a non-negative integer.\n", MY_RANK);
                exit(EXIT_FAILURE);
            }
            REPLICATION_CONF.total_groups = (int)requested_groups;
            break;
        }
        case 'a':
            window_length = atoi(optarg);
            break;
        case 'b':
            merge_offset = atoi(optarg);
            break;
        case 'c':
            VERBOSE = true;
            break;
        case 'd':
            printf("Usage: %s --dataset <dataset> --dataset-size <dataset-size> --queries <queries> --queries-size <query-count> --output <output> --threads <threads> --mode <mode> --threshold <threshold> --k <k> --replication-groups <replication-groups> --window-length <window-length> --merge-offset <merge-offset> [--benchmark-output <path>] [--no-benchmark-report] --verbose\n", argv[0]);
            exit(EXIT_SUCCESS);
        case 'e':
            benchmark_output = optarg;
            break;
        case 'f':
            skip_benchmark_report = true;
            break;
        default:
            printf("Unknown option: %c\n", c);
            exit(EXIT_FAILURE);
        }
    }

    if (dataset == NULL || queries == NULL || dataset_size == 0 || queries_size == 0 || window_length <= 0)
    {
        if (MY_RANK == MASTER)
            printf("[Node %d]: Error: missing required dataset/query/window arguments.\n", MY_RANK);
        exit(EXIT_FAILURE);
    }
    if (window_length % 8 != 0)
    {
        if (MY_RANK == MASTER)
            printf("[Node %d]: Error: SIMD calculations require query length to be a multiple of 8.\n", MY_RANK);
        exit(EXIT_FAILURE);
    }
    if (REPLICATION_CONF.total_groups == 0)
    {
        REPLICATION_CONF.total_groups = COMM_SZ;
    }
    if (REPLICATION_CONF.total_groups > COMM_SZ)
    {
        if (MY_RANK == MASTER)
            printf("[Node %d]: Error: --replication-groups cannot exceed MPI ranks.\n", MY_RANK);
        exit(EXIT_FAILURE);
    }

    rep_init(dataset_size, num_threads);

    data_size_type my_owned_samples = rep_get_time_series_of_group(MY_RANK);
    data_size_type my_samples_to_load = my_owned_samples;
    if (!rep_is_last_node_of_group(MY_RANK))
    {
        my_samples_to_load += (data_size_type)window_length - 1;
    }
    // The r2c FFT has length >= local samples + window - 1 and must fit an int.
    const data_size_type max_local_samples = (data_size_type)INT_MAX - (data_size_type)window_length;
    if (my_samples_to_load > max_local_samples)
    {
        if (MY_RANK == MASTER)
        {
            fprintf(stderr,
                    "[Node %d]: Error: local DMASS chunk has %llu samples, exceeding the 32-bit FFT limit of %llu samples. Increase MPI ranks or reduce dataset size.\n",
                    MY_RANK,
                    my_samples_to_load,
                    max_local_samples);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (MY_RANK == MASTER && VERBOSE)
    {
        printf("---------- Options ----------\n");
        printf("Dataset: %s\n", dataset);
        printf("Dataset Size: %llu\n", dataset_size);
        printf("Queries: %s\n", queries);
        printf("Queries Size: %llu\n", queries_size);
        printf("Output: %s\n", output);
        printf("Threads: %d\n", num_threads);
        printf("Mode: %s\n", mode == MASS_KNN ? "KNN" : "Threshold");
        printf("Verbose: %s\n", VERBOSE ? "True" : "False");
        if (mode == MASS_THRESHOLD)
        {
            printf("Threshold: %f\n", corr_th);
        }
        else
        {
            printf("K: %d\n", k);
        }
        printf("Window Length: %d\n", window_length);
        printf("Merge Offset: %d\n", merge_offset);
        printf("Replication Groups: %d\n", REPLICATION_CONF.total_groups);
        for (int i = 0; i < REPLICATION_CONF.total_groups; i++)
        {
            printf("Replication Group [%d]: \n", REPLICATION_CONF.node_groups[i].group_id);
            printf("    Data Series: %llu\n", REPLICATION_CONF.node_groups[i].total_time_series);
            printf("    Total Nodes: %d\n", REPLICATION_CONF.node_groups[i].total_nodes);
            printf("    Coordinator Node: %d\n", REPLICATION_CONF.node_groups[i].coordinator_node);

            printf("    Nodes: [ ");
            for (int n = 0; n < REPLICATION_CONF.node_groups[i].total_nodes; n++)
            {
                printf("%d ", REPLICATION_CONF.node_groups[i].nodes[n].node_id);
            }
            printf("]\n");

            printf("    Threads: [ ");
            for (int n = 0; n < REPLICATION_CONF.node_groups[i].total_nodes; n++)
            {
                printf("%d ", REPLICATION_CONF.node_groups[i].nodes[n].threads);
            }
            printf("]\n");
        }
        printf("----------------------------\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 3. Load the dataset
    struct timeval index_time_start;
    double index_time;
    gettimeofday(&index_time_start, NULL);
    data_size_type my_total_samples;
    double *my_raw_dataset = load_data(dataset, dataset_size, &my_total_samples, window_length);
    // The data spectrum and prefix sums are query-independent preprocessing,
    // timed with loading like TREMOR's index build.
    mass_ctx_t mass;
    mass_ctx_init(&mass, my_raw_dataset, (int)my_total_samples, window_length, num_threads);
    gettimeofday(&current_time, NULL);
    index_time = (current_time.tv_sec - index_time_start.tv_sec) + (current_time.tv_usec - index_time_start.tv_usec) / 1000000.0;

    MPI_Barrier(MPI_COMM_WORLD);

    // 4. Run scheduling and MASS
    FILE *my_output_file = NULL;
    if (mode == MASS_THRESHOLD && output != NULL)
    {
        char my_output_filename[4096];
        int output_name_len = snprintf(my_output_filename, sizeof(my_output_filename), "%s_%d.csv", output, MY_RANK);
        if (output_name_len < 0 || output_name_len >= (int)sizeof(my_output_filename))
        {
            fprintf(stderr, "[Node %d]: Error: threshold output path is too long for prefix %s\n", MY_RANK, output);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        my_output_file = fopen(my_output_filename, "w");
        if (my_output_file == NULL)
        {
            fprintf(stderr, "[Node %d]: Error: Cannot open file %s: %s\n", MY_RANK, my_output_filename, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    gettimeofday(&qa_time_start, NULL);
    pqueue_bsf **knn_results = static_scheduling(queries, queries_size, window_length, &mass, mode, k, corr_th, merge_offset, my_output_file);
    gettimeofday(&current_time, NULL);
    qa_time = (current_time.tv_sec - qa_time_start.tv_sec) + (current_time.tv_usec - qa_time_start.tv_usec) / 1000000.0;

    // 5. Collect results
    gettimeofday(&collect_time_start, NULL);
    if (mode == MASS_KNN)
    {
        collect_knn_results(knn_results, queries_size, output, merge_offset, window_length, k);
        for (data_size_type i = 0; i < queries_size; i++)
        {
            pqueue_bsf_destroy(knn_results[i]);
        }
        free(knn_results);
    }
    if (my_output_file != NULL)
    {
        if (fflush(my_output_file) != 0 || ferror(my_output_file))
        {
            fprintf(stderr, "[Node %d]: Error: Failed while writing threshold output: %s\n", MY_RANK, strerror(errno));
            fclose(my_output_file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (fclose(my_output_file) != 0)
        {
            fprintf(stderr, "[Node %d]: Error: Failed to close threshold output: %s\n", MY_RANK, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        my_output_file = NULL;
    }
    // Match TREMOR's query phase: search, distributed merge, and result I/O.
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&current_time, NULL);
    qa_time = (current_time.tv_sec - qa_time_start.tv_sec) + (current_time.tv_usec - qa_time_start.tv_usec) / 1000000.0;
    collect_qa_times(qa_time);
    gettimeofday(&current_time, NULL);
    collect_time = (current_time.tv_sec - collect_time_start.tv_sec) + (current_time.tv_usec - collect_time_start.tv_usec) / 1000000.0;

    if (!skip_benchmark_report)
    {
        char benchmark_output_buf[512];
        const char *benchmark_output_path = benchmark_output;
        if (benchmark_output_path == NULL)
        {
            if (output != NULL)
            {
                snprintf(benchmark_output_buf, sizeof(benchmark_output_buf), "%s.timings.csv", output);
            }
            else
            {
                snprintf(benchmark_output_buf, sizeof(benchmark_output_buf), "dmass_timings.csv");
            }
            benchmark_output_path = benchmark_output_buf;
        }
        collect_and_write_timings(index_time, qa_time, benchmark_output_path);
    }

    mass_ctx_destroy(&mass);
    free(my_raw_dataset);

    rep_destroy();
    MPI_Finalize();

    gettimeofday(&current_time, NULL);
    total_time = (current_time.tv_sec - total_time_start.tv_sec) + (current_time.tv_usec - total_time_start.tv_usec) / 1000000.0;

    if (MY_RANK == MASTER && VERBOSE)
        printf("[Node %d]: Total Time: %lfs. Collect Time: %lfs\n", MY_RANK, total_time, collect_time);

    return 0;
}

void rep_init(data_size_type dataset_size, int num_threads)
{
    rep_group_t *node_groups = (rep_group_t *)malloc(sizeof(rep_group_t) * REPLICATION_CONF.total_groups);
    int *node_group_mappings = (int *)malloc(sizeof(int) * COMM_SZ);

    int nodes_of_each_group = COMM_SZ / REPLICATION_CONF.total_groups;
    for (int i = 0; i < REPLICATION_CONF.total_groups; i++)
    {
        int nodes_of_this_group;
        if (i == REPLICATION_CONF.total_groups - 1)
        {
            nodes_of_this_group = COMM_SZ - nodes_of_each_group * (REPLICATION_CONF.total_groups - 1);
        }
        else
        {
            nodes_of_this_group = nodes_of_each_group;
        }

        node_groups[i].total_nodes = nodes_of_this_group;
        node_groups[i].group_id = i;

        node_groups[i].nodes = (system_node_t *)malloc(sizeof(system_node_t) * nodes_of_this_group);

        for (int node = 0; node < nodes_of_this_group; node++)
        {
            node_groups[i].nodes[node].node_id = i * nodes_of_each_group + node;
            node_groups[i].nodes[node].threads = num_threads;

            if (node == 0)
            {
                node_groups[i].coordinator_node = node_groups[i].nodes[node].node_id;
            }

            node_group_mappings[node_groups[i].nodes[node].node_id] = i;
        }

        node_groups[i].total_time_series = rep_allocate_data_series_default(i, dataset_size);
    }

    REPLICATION_CONF.node_groups = node_groups;
    REPLICATION_CONF.node_group_mappings = node_group_mappings;

    data_size_type time_series_sum = 0;
    int nodes_sum = 0;
    for (int i = 0; i < REPLICATION_CONF.total_groups; i++)
    {
        nodes_sum += node_groups[i].total_nodes;
        time_series_sum += node_groups[i].total_time_series;
    }

    if (time_series_sum != dataset_size)
    {
        if (MY_RANK == MASTER)
            printf("Error: The sum of the data series allocated to each group is not equal to the whole dataset size.\n");

        exit(EXIT_FAILURE);
    }

    if (COMM_SZ != nodes_sum)
    {
        if (MY_RANK == MASTER)
            printf("Error: The sum of the nodes is different from total nodes\n");

        exit(EXIT_FAILURE);
    }

    return;
}

void rep_destroy()
{
    for (int i = 0; i < REPLICATION_CONF.total_groups; i++)
    {
        free(REPLICATION_CONF.node_groups[i].nodes);
    }

    free(REPLICATION_CONF.node_groups);
    free(REPLICATION_CONF.node_group_mappings);
}

data_size_type rep_allocate_data_series_default(int group_id, data_size_type dataset_size)
{
    int chunks = REPLICATION_CONF.total_groups;
    data_size_type chunk_size_aprox = dataset_size / REPLICATION_CONF.total_groups;
    double chunk_size = 1.0 * dataset_size / REPLICATION_CONF.total_groups;

    if (group_id != chunks - 1)
    {
        return chunk_size_aprox;
    }

    return dataset_size - chunk_size_aprox * (chunks - 1);
}

int rep_find_group(int rank)
{
    return REPLICATION_CONF.node_group_mappings[rank];
}

bool rep_is_last_node_of_group(int rank)
{
    return rank == (REPLICATION_CONF.node_groups[rep_find_group(rank)].nodes[REPLICATION_CONF.node_groups[rep_find_group(rank)].total_nodes - 1].node_id);
}

int rep_find_coordinator_node_rank(int rank)
{
    return REPLICATION_CONF.node_groups[rep_find_group(rank)].coordinator_node;
}

int rep_get_repgroup_nodes(int rank)
{
    return REPLICATION_CONF.node_groups[rep_find_group(rank)].total_nodes;
}

data_size_type rep_get_time_series_of_group(int rank)
{
    return REPLICATION_CONF.node_groups[rep_find_group(rank)].total_time_series;
}

rep_group_t rep_get_group(int rank)
{
    return REPLICATION_CONF.node_groups[rep_find_group(rank)];
}

data_size_type rep_get_time_series_offset(int rank)
{
    int group_id = rep_find_group(rank);
    data_size_type offset = 0;

    for (int i = 0; i < group_id; i++)
    {
        offset += REPLICATION_CONF.node_groups[i].total_time_series;
    }

    return offset;
}

// Data Partitioning
double *load_data(char *ifilename, data_size_type total_samples, data_size_type *loaded_samples, int window_size)
{
    FILE *ifile = fopen(ifilename, "rb");
    if (ifile == NULL)
    {
        fprintf(stderr, "File %s not found!\n", ifilename);
        exit(EXIT_FAILURE);
    }

    fseek(ifile, 0L, SEEK_END);
    file_position_type sz = (file_position_type)ftell(ifile);
    fseek(ifile, 0L, SEEK_SET);

    file_position_type total_records = sz / sizeof(float); // calculate how many samples the file has
    if (total_records < total_samples)
    {
        fprintf(stderr, "File %s has only %llu records!\n", ifilename, total_records);
        exit(EXIT_FAILURE);
    }

    if (MY_RANK == MASTER)
    {
        printf("[Node %d]: File %s has %llu records in total.\n", MY_RANK, ifilename, total_records);
    }

    data_size_type total_samples_of_my_group = rep_get_time_series_of_group(MY_RANK);
    data_size_type possition_to_file = rep_get_time_series_offset(MY_RANK);
    int my_node_group_id = rep_find_group(MY_RANK);
    int returned_val = fseek(ifile, possition_to_file * sizeof(float), SEEK_SET);
    if (returned_val != 0)
    {
        printf("Error on fseek()\n");
        exit(EXIT_FAILURE);
    }
    data_size_type total_samples_to_load = (my_node_group_id == REPLICATION_CONF.total_groups - 1) ? total_samples_of_my_group : total_samples_of_my_group + window_size - 1;
    if (total_samples_to_load > (data_size_type)INT_MAX - (data_size_type)window_size || total_samples_to_load < (data_size_type)window_size)
    {
        fprintf(stderr, "Invalid local chunk length %llu for FFTW/window length\n", total_samples_to_load);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    float *rawfile = (float *)malloc(sizeof(float) * total_samples_to_load);
    size_t bytes_read = fread(rawfile, sizeof(float), total_samples_to_load, ifile);
    if (bytes_read != total_samples_to_load)
    {
        fprintf(stderr, "Error: Read %zu records, expected %llu.\n", bytes_read, total_samples_to_load);
        exit(EXIT_FAILURE);
    }

    *loaded_samples = total_samples_to_load;

    printf("[Node %d]: Read %zu records, starting from %llu. Total samples to load: %llu\n", MY_RANK, bytes_read, possition_to_file, total_samples_to_load);

    double *d_rawfile = (double *)malloc(sizeof(double) * total_samples_to_load);
    for (int i = 0; i < total_samples_to_load; i++)
    {
        d_rawfile[i] = (double)rawfile[i];
    }
    free(rawfile);
    fclose(ifile);

    return d_rawfile;
}

// Scheduling
pqueue_bsf **static_scheduling(const char *ifilename, data_size_type q_num, int window_size,
                               mass_ctx_t *mass, MASS_MODE mode, int k, float threshold,
                               int merge_offset, FILE *my_output_file)
{
    if (mode == MASS_THRESHOLD && my_output_file != NULL)
    {
        fprintf(my_output_file, "Query, Counter, Position, Cross Correlation\n");
    }

    data_size_type sets, from, to;
    sets = q_num / rep_get_repgroup_nodes(MY_RANK);
    from = (MY_RANK - rep_find_coordinator_node_rank(MY_RANK)) * sets;
    if (rep_is_last_node_of_group(MY_RANK))
    {
        to = q_num;
    }
    else
    {
        to = ((MY_RANK - rep_find_coordinator_node_rank(MY_RANK)) + 1) * sets;
    }

    if (VERBOSE)
        printf("[Node %d]: Answering from %d to %d\n", MY_RANK, from, to);

    FILE *ifile = fopen(ifilename, "rb");
    if (ifile == NULL)
    {
        fprintf(stderr, "File %s not found!\n", ifilename);
        exit(EXIT_FAILURE);
    }

    double **queries = (double **)malloc(sizeof(double *) * q_num);
    float *curr_query = (float *)malloc(sizeof(float) * window_size);
    for (data_size_type i = 0; i < q_num; i++)
    {
        queries[i] = (double *)malloc(sizeof(double) * window_size);

        if (fread(curr_query, sizeof(float), window_size, ifile) != (size_t)window_size)
        {
            fprintf(stderr, "Truncated query file %s\n", ifilename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int j = 0; j < window_size; j++)
        {
            queries[i][j] = (double)curr_query[j];
        }
    }
    free(curr_query);
    fclose(ifile);

    pqueue_bsf **knn_pqueues = NULL;
    if (mode == MASS_KNN)
    {
        knn_pqueues = (pqueue_bsf **)malloc(sizeof(pqueue_bsf *) * q_num);
        for (int i = 0; i < q_num; i++)
        {
            knn_pqueues[i] = pqueue_bsf_init(merge_offset > 0 ? 2 * k : k);
        }
        if (merge_offset > 0)
            KNN_CANDIDATES = (knn_candidates_t *)calloc(q_num, sizeof(knn_candidates_t));
    }

    int n = mass->n;
    int m = (int)window_size;
    double *dist = (double *)malloc(sizeof(double) * n);
    for (data_size_type qi = from; qi < to; qi++)
    {
        double query_time;
        struct timeval query_start, query_end;
        gettimeofday(&query_start, NULL);
        mass_ctx_query(mass, queries[qi], dist);

        // post process the results
        if (mode == MASS_KNN)
        {
            for (int i = 0; i < n - m + 1; i++)
            {
                const float d = (float)dist[i];
                const float bound = knn_pqueues[qi]->knn[knn_pqueues[qi]->k - 1];
                if (d <= bound)
                {
                    file_position_type global_pos = i + rep_get_time_series_offset(MY_RANK);
                    if (KNN_CANDIDATES != NULL)
                        knn_candidates_add(&KNN_CANDIDATES[qi], global_pos, d);
                    if (d < bound)
                        pqueue_bsf_insert_offset(knn_pqueues[qi], d, global_pos, merge_offset);
                }
            }
        }
        else if (mode == MASS_THRESHOLD)
        {
            if (my_output_file != NULL)
            {
                data_size_type counter = 0;
                for (int i = 0; i < n - m + 1; i++)
                {
                    double corr = 1 - dist[i] / (2 * m);

                    if (corr >= threshold && my_output_file != NULL)
                    {
                        file_position_type global_pos = i + rep_get_time_series_offset(MY_RANK);
                        fprintf(my_output_file, "%llu, %llu, %llu, %f\n", qi, counter, global_pos, corr);

                        i += merge_offset;
                        counter++;
                    }
                }
            }
        }

        gettimeofday(&query_end, NULL);
        query_time = (query_end.tv_sec - query_start.tv_sec) + (query_end.tv_usec - query_start.tv_usec) / 1000000.0;
        if (ENABLE_PER_QUERY_PRINT && VERBOSE)
            printf("[Node %d]: Query %lu: %lfs\n", MY_RANK, qi, query_time);
    }

    for (data_size_type i = 0; i < q_num; i++)
    {
        free(queries[i]);
    }
    free(queries);
    free(dist);

    MPI_Barrier(MPI_COMM_WORLD);

    return knn_pqueues;
}

// MASS
// The local chunk's spectrum and prefix sums are computed once per rank.
// Each query then costs one r2c and one c2r FFT whose length is padded to a
// 7-smooth size, so runtime tracks the chunk length rather than its prime
// factorisation.
static long long next_smooth_length(long long target)
{
    long long best = LLONG_MAX;
    for (long long p2 = 1; p2 < best; p2 *= 2)
        for (long long p3 = p2; p3 < best; p3 *= 3)
            for (long long p5 = p3; p5 < best; p5 *= 5)
                for (long long p7 = p5; p7 < best; p7 *= 7)
                    if (p7 >= target)
                    {
                        best = p7;
                        break;
                    }
    return best;
}

void mass_ctx_init(mass_ctx_t *ctx, double *x, int n, int m, int num_threads)
{
    ctx->x = x;
    ctx->n = n;
    ctx->m = m;
    ctx->num_threads = num_threads;

    // Linear correlation needs n + m - 1 points to avoid circular wrap-around.
    long long fft_len = next_smooth_length((long long)n + m - 1);
    if (fft_len > INT_MAX)
    {
        fprintf(stderr, "[Node %d]: Error: FFT length %lld exceeds INT_MAX\n", MY_RANK, fft_len);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    ctx->fft_len = (int)fft_len;
    const int bins = ctx->fft_len / 2 + 1;

    ctx->buf = fftw_alloc_real(ctx->fft_len);
    ctx->X = fftw_alloc_complex(bins);
    ctx->Y = fftw_alloc_complex(bins);
    const int nwindows = n - m + 1;
    ctx->mu = (double *)malloc(sizeof(double) * (size_t)nwindows);
    ctx->sigma = (double *)malloc(sizeof(double) * (size_t)nwindows);
    if (ctx->buf == NULL || ctx->X == NULL || ctx->Y == NULL || ctx->mu == NULL || ctx->sigma == NULL)
    {
        fprintf(stderr, "[Node %d]: Error: cannot allocate MASS buffers for FFT length %d\n", MY_RANK, ctx->fft_len);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fftw_init_threads();
    fftw_plan_with_nthreads(num_threads);
    // FFTW_ESTIMATE planning does not touch the arrays, so plan before filling.
    fftw_plan data_forward = fftw_plan_dft_r2c_1d(ctx->fft_len, ctx->buf, ctx->X, FFTW_ESTIMATE);
    ctx->query_forward = fftw_plan_dft_r2c_1d(ctx->fft_len, ctx->buf, ctx->Y, FFTW_ESTIMATE);
    ctx->query_backward = fftw_plan_dft_c2r_1d(ctx->fft_len, ctx->Y, ctx->buf, FFTW_ESTIMATE);

    double *buf = ctx->buf;
    const int len = ctx->fft_len;
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; i++)
    {
        buf[i] = (i < n && x[i] != MISSING_VALUE) ? x[i] : 0.0;
    }
    fftw_execute(data_forward);
    fftw_destroy_plan(data_forward);

    // Window statistics are computed exactly (two passes per window). Global
    // prefix sums of x^2 lose every significant digit in near-silent windows
    // (gaps, taper edges) next to large-amplitude data, which reported
    // correlation 1.0 for windows whose true correlation was ~0.2.
    double sumsq = 0.0;
#pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+ : sumsq)
    for (int j = 0; j < nwindows; j++)
    {
        const double *w = x + j;
        double s = 0.0;
        for (int i = 0; i < m; i++)
            s += (w[i] != MISSING_VALUE) ? w[i] : 0.0;
        const double mean = s / m;
        double ss = 0.0;
        for (int i = 0; i < m; i++)
        {
            const double d = ((w[i] != MISSING_VALUE) ? w[i] : 0.0) - mean;
            ss += d * d;
        }
        ctx->mu[j] = mean;
        ctx->sigma[j] = sqrt(ss / m);
        const double v = (x[j] != MISSING_VALUE) ? x[j] : 0.0;
        sumsq += v * v;
    }
    for (int i = nwindows; i < n; i++)
    {
        const double v = (x[i] != MISSING_VALUE) ? x[i] : 0.0;
        sumsq += v * v;
    }
    // FFT correlation has absolute error ~ eps * log2(len) * ||x|| * ||y||, with
    // ||y|| = sqrt(m) for a z-normalised query. Windows whose m * sigma is
    // within 1e4 of that bound get an exact dot product, so correlation
    // errors stay below ~1e-4.
    ctx->fft_dot_min = 1e4 * DBL_EPSILON * log2((double)ctx->fft_len) * sqrt(sumsq) * sqrt((double)m);

    if (VERBOSE)
        printf("[Node %d]: MASS FFT length %d for %d local samples\n", MY_RANK, ctx->fft_len, n);
}

void mass_ctx_destroy(mass_ctx_t *ctx)
{
    fftw_destroy_plan(ctx->query_forward);
    fftw_destroy_plan(ctx->query_backward);
    fftw_free(ctx->buf);
    fftw_free(ctx->X);
    fftw_free(ctx->Y);
    free(ctx->mu);
    free(ctx->sigma);
    fftw_cleanup_threads();
}

// Fills dist[0 .. n-m] with z-normalised squared Euclidean distances; windows
// that are missing or constant get FLT_MAX.
void mass_ctx_query(mass_ctx_t *ctx, double *y, double *dist)
{
    const int n = ctx->n;
    const int m = ctx->m;
    const int len = ctx->fft_len;
    const int bins = len / 2 + 1;
    const int nwindows = n - m + 1;
    const int num_threads = ctx->num_threads;
    const double *x = ctx->x;
    const double *mu = ctx->mu;
    const double *sigma = ctx->sigma;
    const double fft_dot_min = ctx->fft_dot_min;
    double *buf = ctx->buf;
    fftw_complex *X = ctx->X;
    fftw_complex *Y = ctx->Y;

    y = zNorm(y, m, y);
    double sumy = 0.0, sumy2 = 0.0;
    for (int i = 0; i < m; i++)
    {
        sumy += y[i];
        sumy2 += y[i] * y[i];
    }
    const double meany = sumy / m;
    const double sigmay = sqrt((sumy2 / m) - meany * meany);
    if (!isfinite(sigmay) || sigmay <= 1e-12)
    {
#pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int j = 0; j < nwindows; j++)
            dist[j] = FLT_MAX;
        return;
    }

#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; i++)
    {
        buf[i] = (i < m) ? y[m - 1 - i] : 0.0; // reversed query
    }
    fftw_execute(ctx->query_forward);

#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < bins; i++)
    {
        const double re = X[i][0] * Y[i][0] - X[i][1] * Y[i][1];
        const double im = X[i][1] * Y[i][0] + X[i][0] * Y[i][1];
        Y[i][0] = re;
        Y[i][1] = im;
    }
    fftw_execute(ctx->query_backward);

    const double scale = 1.0 / len;
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int j = 0; j < nwindows; j++)
    {
        if (x[j] == MISSING_VALUE)
        {
            dist[j] = FLT_MAX;
            continue;
        }

        const double meanx = mu[j];
        const double sigmax = sigma[j];
        if (!isfinite(sigmax) || sigmax <= 1e-12)
        {
            dist[j] = FLT_MAX;
            continue;
        }
        double sumxy = buf[m - 1 + j] * scale;
        if (m * sigmax < fft_dot_min)
        {
            sumxy = 0.0;
            for (int i = 0; i < m; i++)
                sumxy += y[i] * ((x[j + i] != MISSING_VALUE) ? x[j + i] : 0.0);
        }

        double c = (sumxy - m * meanx * meany) / (m * sigmax * sigmay);
        c = fmax(-1.0, fmin(1.0, c));
        dist[j] = 2 * m * (1 - c);
    }
}

double *zNorm(double *x, int n, double *y)
{
    double ex = 0, ex2 = 0;
    for (int i = 0; i < n; i++)
    {
        ex += x[i];
        ex2 += x[i] * x[i];
    }
    double mean = ex / n;
    double std = ex2 / n;
    std = sqrt(std - mean * mean);
    for (int i = 0; i < n; i++)
        y[i] = (x[i] - mean) / std;
    return y;
}

pqueue_bsf *pqueue_bsf_init(int k)
{
    pqueue_bsf *q;

    if (!(q = (pqueue_bsf *)malloc(sizeof(pqueue_bsf))))
        return NULL;
    if (!(q->position = (file_position_type *)malloc(sizeof(file_position_type) * k)))
        return NULL;
    if (!(q->knn = (float *)malloc(sizeof(float) * k)))
        return NULL;

    for (int i = 0; i < k; ++i)
    {
        q->knn[i] = FLT_MAX;
        q->position[i] = ULLONG_MAX;
    }

    q->k = k;
    return q;
}

void pqueue_bsf_destroy(pqueue_bsf *q)
{
    free(q->position);
    free(q->knn);
    free(q);
}

/*
 * Exclusion-zone insert for kNN with a merge offset: entries are kept pairwise
 * at least pos_offset positions apart. A window closer than pos_offset to an
 * entry that is at least as good is rejected; otherwise it replaces every entry
 * within pos_offset (all worse) and is inserted in distance order. TREMOR
 * implements the same rule.
 */
static int within_offset(file_position_type a, file_position_type b, int pos_offset)
{
    return (a > b ? a - b : b - a) < (file_position_type)pos_offset;
}

void pqueue_bsf_insert_offset(pqueue_bsf *q, float data, file_position_type position, int pos_offset)
{
    if (pos_offset == 0)
    {
        pqueue_bsf_insert(q, data, position);
        return;
    }

    for (int i = 0; i < q->k; i++)
    {
        if (q->knn[i] < FLT_MAX && within_offset(q->position[i], position, pos_offset) && q->knn[i] <= data)
            return;
    }

    int kept = 0;
    for (int i = 0; i < q->k; i++)
    {
        if (q->knn[i] < FLT_MAX && within_offset(q->position[i], position, pos_offset))
            continue;
        q->knn[kept] = q->knn[i];
        q->position[kept] = q->position[i];
        kept++;
    }
    for (int i = kept; i < q->k; i++)
    {
        q->knn[i] = FLT_MAX;
        q->position[i] = ULLONG_MAX;
    }

    for (int i = 0; i < q->k; i++)
    {
        if (data <= q->knn[i])
        {
            for (int j = q->k - 1; j > i; j--)
            {
                q->knn[j] = q->knn[j - 1];
                q->position[j] = q->position[j - 1];
            }
            q->knn[i] = data;
            q->position[i] = position;
            return;
        }
    }
}

void knn_candidates_add(knn_candidates_t *list, file_position_type position, float distance)
{
    if (list->count == list->capacity)
    {
        list->capacity = list->capacity ? 2 * list->capacity : 1024;
        list->items = (knn_candidate_t *)realloc(list->items, sizeof(knn_candidate_t) * list->capacity);
        if (list->items == NULL)
        {
            fprintf(stderr, "[Node %d]: Error: out of memory for kNN candidates\n", MY_RANK);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    list->items[list->count].position = position;
    list->items[list->count].distance = distance;
    list->count++;
}

static int compare_knn_candidates(const void *pa, const void *pb)
{
    const knn_candidate_t *a = (const knn_candidate_t *)pa, *b = (const knn_candidate_t *)pb;
    if (a->distance != b->distance)
        return a->distance < b->distance ? -1 : 1;
    return a->position < b->position ? -1 : (a->position > b->position ? 1 : 0);
}

/* Canonical exclusion-zone selection; sorts `candidates`. Returns the number of answers. */
int select_knn_with_exclusion(knn_candidate_t *candidates, size_t count, int k, int merge_offset, knn_candidate_t *answer)
{
    qsort(candidates, count, sizeof(knn_candidate_t), compare_knn_candidates);
    file_position_type *kept = (file_position_type *)malloc(sizeof(file_position_type) * (k > 0 ? k : 1)); // sorted
    int n = 0;
    for (size_t c = 0; c < count && n < k; c++)
    {
        const file_position_type p = candidates[c].position;
        int lo = 0, hi = n; // first kept position >= p
        while (lo < hi)
        {
            const int mid = (lo + hi) / 2;
            if (kept[mid] < p)
                lo = mid + 1;
            else
                hi = mid;
        }
        if ((lo < n && within_offset(kept[lo], p, merge_offset)) || (lo > 0 && within_offset(kept[lo - 1], p, merge_offset)))
            continue;
        memmove(&kept[lo + 1], &kept[lo], sizeof(file_position_type) * (n - lo));
        kept[lo] = p;
        answer[n++] = candidates[c];
    }
    free(kept);
    return n;
}

void pqueue_bsf_insert(pqueue_bsf *q, float data, file_position_type position)
{
    int i, j;

    for (i = 0; i < q->k; i++)
    {
        if (data <= q->knn[i] && q->position[i] != position) // todo!=> check if this bug-free
        {
            for (j = q->k - 1; j > i; j--)
            {
                q->knn[j] = q->knn[j - 1];
                q->position[j] = q->position[j - 1];
            }

            q->knn[i] = data;
            q->position[i] = position;

            return;
        }
    }
}

// Collect
void collect_knn_results(pqueue_bsf **knn_pqueues, data_size_type queries_size, char *output, int merge_offset, int window_size, int k)
{
    if (merge_offset > 0)
    {
        // Candidates within each rank's final bound, gathered on the master.
        int *counts = (int *)malloc(sizeof(int) * queries_size);
        size_t local_total = 0;
        for (data_size_type q = 0; q < queries_size; q++)
        {
            const float bound = knn_pqueues[q]->knn[knn_pqueues[q]->k - 1];
            knn_candidates_t *list = &KNN_CANDIDATES[q];
            size_t kept = 0;
            for (size_t c = 0; c < list->count; c++)
            {
                if (list->items[c].distance <= bound)
                    list->items[kept++] = list->items[c];
            }
            list->count = kept;
            counts[q] = (int)kept;
            local_total += kept;
        }
        unsigned long long *positions = (unsigned long long *)malloc(sizeof(unsigned long long) * (local_total ? local_total : 1));
        float *distances = (float *)malloc(sizeof(float) * (local_total ? local_total : 1));
        size_t at = 0;
        for (data_size_type q = 0; q < queries_size; q++)
        {
            for (size_t c = 0; c < KNN_CANDIDATES[q].count; c++, at++)
            {
                positions[at] = KNN_CANDIDATES[q].items[c].position;
                distances[at] = KNN_CANDIDATES[q].items[c].distance;
            }
            free(KNN_CANDIDATES[q].items);
        }
        free(KNN_CANDIDATES);
        KNN_CANDIDATES = NULL;

        const int total = (int)local_total;
        int *all_counts = NULL, *totals = NULL, *displs = NULL;
        unsigned long long *all_positions = NULL;
        float *all_distances = NULL;
        if (MY_RANK == MASTER)
        {
            all_counts = (int *)malloc(sizeof(int) * queries_size * COMM_SZ);
            totals = (int *)malloc(sizeof(int) * COMM_SZ);
            displs = (int *)malloc(sizeof(int) * COMM_SZ);
        }
        MPI_Gather(counts, (int)queries_size, MPI_INT, all_counts, (int)queries_size, MPI_INT, MASTER, MPI_COMM_WORLD);
        MPI_Gather(&total, 1, MPI_INT, totals, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        size_t grand_total = 0;
        if (MY_RANK == MASTER)
        {
            for (int r = 0; r < COMM_SZ; r++)
            {
                displs[r] = (int)grand_total;
                grand_total += (size_t)totals[r];
            }
            all_positions = (unsigned long long *)malloc(sizeof(unsigned long long) * (grand_total ? grand_total : 1));
            all_distances = (float *)malloc(sizeof(float) * (grand_total ? grand_total : 1));
        }
        MPI_Gatherv(positions, total, MPI_UNSIGNED_LONG_LONG, all_positions, totals, displs, MPI_UNSIGNED_LONG_LONG, MASTER, MPI_COMM_WORLD);
        MPI_Gatherv(distances, total, MPI_FLOAT, all_distances, totals, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
        free(positions);
        free(distances);
        free(counts);
        if (MY_RANK != MASTER)
            return;

        FILE *of = NULL;
        if (output)
        {
            of = fopen(output, "w+");
            if (!of)
            {
                fprintf(stderr, "Error opening output file %s\n", output);
                exit(EXIT_FAILURE);
            }
            fprintf(of, "tID, k, pos, corr\n");
        }
        // Per query: collect its candidates from every rank's block.
        size_t *rank_offset = (size_t *)calloc(COMM_SZ, sizeof(size_t));
        knn_candidate_t *query_candidates = (knn_candidate_t *)malloc(sizeof(knn_candidate_t) * (grand_total ? grand_total : 1));
        knn_candidate_t *answer = (knn_candidate_t *)malloc(sizeof(knn_candidate_t) * k);
        for (data_size_type q = 0; q < queries_size; q++)
        {
            size_t n = 0;
            for (int r = 0; r < COMM_SZ; r++)
            {
                const int c = all_counts[(size_t)r * queries_size + q];
                for (int j = 0; j < c; j++)
                {
                    const size_t idx = (size_t)displs[r] + rank_offset[r] + j;
                    query_candidates[n].position = all_positions[idx];
                    query_candidates[n].distance = all_distances[idx];
                    n++;
                }
                rank_offset[r] += (size_t)c;
            }
            const int found = select_knn_with_exclusion(query_candidates, n, k, merge_offset, answer);
            if (of)
            {
                for (int ik = 0; ik < k; ik++)
                {
                    const unsigned long long pos = ik < found ? answer[ik].position : 0ULL;
                    const float d = ik < found ? answer[ik].distance : FLT_MAX;
                    fprintf(of, "%d, %d, %llu, %f\n", (int)q, ik + 1, pos, 1 - d / (2 * window_size));
                }
            }
        }
        if (of)
            fclose(of);
        free(rank_offset);
        free(query_candidates);
        free(answer);
        free(all_counts);
        free(totals);
        free(displs);
        free(all_positions);
        free(all_distances);
        return;
    }

    int topk = knn_pqueues[0]->k;

    if (MY_RANK != MASTER)
    {
        MPI_Request *requests = (MPI_Request *)malloc(sizeof(MPI_Request) * 2 * queries_size);

        for (int i = 0; i < queries_size; i++)
        {
            float *sel_knn = knn_pqueues[i]->knn;
            file_position_type *sel_pos = knn_pqueues[i]->position;

            MPI_Isend(sel_knn, topk, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &requests[2 * i]);
            MPI_Isend(sel_pos, topk, MPI_UNSIGNED_LONG_LONG, MASTER, 0, MPI_COMM_WORLD, &requests[2 * i + 1]);
        }

        MPI_Waitall(2 * queries_size, requests, MPI_STATUSES_IGNORE);

        free(requests);
        return;
    }

    float *node_knns = (float *)malloc(sizeof(float) * topk);
    file_position_type *node_positions = (file_position_type *)malloc(sizeof(file_position_type) * topk);

    FILE *of = NULL;
    if (output)
    {
        of = fopen(output, "w+");
        if (!of)
        {
            fprintf(stderr, "Error opening output file %s\n", output);
            exit(EXIT_FAILURE);
        }
        fprintf(of, "tID, k, pos, corr\n");
    }

    for (int i = 0; i < queries_size; i++)
    {
        pqueue_bsf *final_result_pq = pqueue_bsf_init(topk);

        for (int j = 0; j < COMM_SZ; j++)
        {
            if (j == MASTER)
            {

                for (int k = 0; k < topk; k++)
                {
                    pqueue_bsf_insert_offset(final_result_pq, knn_pqueues[i]->knn[k], knn_pqueues[i]->position[k], merge_offset);
                }
            }
            else
            {
                MPI_Recv(node_knns, topk, MPI_FLOAT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(node_positions, topk, MPI_UNSIGNED_LONG_LONG, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // printf("[Node %d]: Received knn and positions of node %d\n", my_rank, j);
                if (node_knns[0] < final_result_pq->knn[topk - 1])
                {
                    for (int k = 0; k < topk; k++)
                    {
                        pqueue_bsf_insert_offset(final_result_pq, node_knns[k], node_positions[k], merge_offset);
                    }
                }
            }
        }

        if (output)
        {
            for (int ik = 0; ik < topk; ik++)
            {
                float corr = 1 - final_result_pq->knn[ik] / (2 * window_size);

                fprintf(of, "%d, %d, %llu, %f\n", i, ik + 1, final_result_pq->position[ik], corr);
            }
        }

        pqueue_bsf_destroy(final_result_pq);
    }

    if (output)
    {
        fclose(of);
    }

    // free all
    /*for (int i = 0; i < queries_size; i++)
    {
        free(query_times[i]);
    }
    free(query_times);
    free(node_knns);
    free(node_positions);
    free(per_query_times);*/
}

void collect_qa_times(double qa_time)
{
    if (MY_RANK != MASTER)
    {
        MPI_Send(&qa_time, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
        return;
    }

    double *node_qa_times = (double *)malloc(sizeof(double) * COMM_SZ);
    node_qa_times[MASTER] = qa_time;

    for (int i = 0; i < COMM_SZ; i++)
    {
        if (i == MASTER)
        {
            continue;
        }

        MPI_Recv(&node_qa_times[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (VERBOSE)
    {
        for (int i = 0; i < COMM_SZ; i++)
        {
            printf("[Node %d]: QA Time: %lfs\n", i, node_qa_times[i]);
        }

        // Find max time
        double max_time = 0;
        for (int i = 0; i < COMM_SZ; i++)
        {
            if (node_qa_times[i] > max_time)
            {
                max_time = node_qa_times[i];
            }
        }

        // printf("Max QA Time: %lfs\n", max_time);
    }

    free(node_qa_times);
}

void collect_and_write_timings(double index_time, double query_time, const char *path)
{
    double *all_index_times = NULL;
    double *all_query_times = NULL;
    if (MY_RANK == MASTER)
    {
        all_index_times = (double *)malloc(sizeof(double) * COMM_SZ);
        all_query_times = (double *)malloc(sizeof(double) * COMM_SZ);
    }

    MPI_Gather(&index_time, 1, MPI_DOUBLE, all_index_times, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Gather(&query_time, 1, MPI_DOUBLE, all_query_times, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    if (MY_RANK != MASTER)
    {
        return;
    }

    if (path == NULL)
    {
        free(all_index_times);
        free(all_query_times);
        return;
    }

    FILE *of = fopen(path, "w");
    if (of == NULL)
    {
        fprintf(stderr, "Warning: cannot open benchmark report file %s\n", path);
        free(all_index_times);
        free(all_query_times);
        return;
    }

    fprintf(of, "node_rank,index_time,query_time\n");
    for (int r = 0; r < COMM_SZ; r++)
    {
        fprintf(of, "%d,%.6f,%.6f\n", r, all_index_times[r], all_query_times[r]);
    }
    fclose(of);

    if (VERBOSE)
    {
        printf("[Node %d]: Benchmark timings CSV: %s\n", MY_RANK, path);
    }

    free(all_index_times);
    free(all_query_times);
}
