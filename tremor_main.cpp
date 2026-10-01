#include "tremor/algos/htremor/Tremor.hpp"
#include "tremor/algos/sss/SkipSequential.hpp"

#include <mpi.h>

#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    enum class SearchMode
    {
        KNN,
        Threshold,
    };

    struct DriverOptions
    {
        std::string dataset_path;
        std::string queries_path;
        std::string output_path;
        std::string benchmark_output_path;

        bool dataset_size_set = false;
        bool query_count_set = false;
        bool skip_benchmark_report = false;
        idx_t dataset_size = 0;
        idx_t query_count = 0;
        idx_t window_length = 0;

        SearchMode mode = SearchMode::KNN;
        idx_t knn_k = 10;
        idx_t threshold_result_slots = 1;
        float threshold_correlation = 0.93f;

        int search_workers = 4;
        int index_threads = 4;
        int leaf_size = 2000;
        int paa_segments = 16;
        int replication_groups = 0;
        int pq_threshold_divisor = 16;
        int merge_offset = 0;
        RefinementMode refinement = RefinementMode::Leaf;
        bool skip_sequential = false; // --method skip-sequential or sequential: the baselines without an index tree
        bool lower_bounds = true;     // false for --method sequential (no iSAX summaries, no lower bounds)
        double scan_fallback_fraction = 0.01;
        bool verbose = false;

        bool print_knn_results = false;
        idx_t print_query_limit = 5;
    };

    struct BenchmarkTimings
    {
        double index_build_seconds = 0.0;
        double search_seconds = 0.0;
    };

    struct HelpRequested
    {
    };

    void print_usage(const char *program)
    {
        std::printf(
            "Usage:\n"
            "  %s --dataset <waveform.bin> --queries <queries.bin> --window-length <N> [options]\n"
            "\n"
            "Required inputs:\n"
            "  --dataset PATH              Raw float32 long sequence file.\n"
            "  --queries PATH              Raw float32 query/template file.\n"
            "  --window-length N           Number of float32 samples per query/subsequence.\n"
            "\n"
            "Input sizing:\n"
            "  --dataset-size N            Number of float32 samples to read from dataset.\n"
            "                              Default: infer from dataset file size.\n"
            "  --query-count N             Number of query vectors in --queries.\n"
            "                              Default: infer from query file size / window length.\n"
            "\n"
            "Search mode:\n"
            "  --mode knn                  Run k-nearest-neighbor search (default).\n"
            "  --mode threshold            Run threshold/correlation search.\n"
            "  --k N                       K for KNN mode (default: 10).\n"
            "  --threshold X               Correlation threshold for threshold mode (default: 0.93).\n"
            "  --threshold-result-slots N   Scratch result slots passed to threshold search (default: 1).\n"
            "  --method tremor|skip-sequential|sequential\n"
            "                              tremor (default), or the skip-sequential scan baseline:\n"
            "                              no index tree, subsequences scanned in waveform order with\n"
            "                              the iSAX lower bound and early abandoning; sequential: the\n"
            "                              same scan with early abandoning only (no lower bounds).\n"
            "  --refinement leaf|adaptive  How candidates get exact distances (default: leaf).\n"
            "                              leaf: leaf by leaf in lower-bound order (classic TREMOR).\n"
            "                              adaptive (TREMOR, as in the paper): per query, the index\n"
            "                              when few windows of a sample pass the lower bound; else a\n"
            "                              scan in waveform order (with the lower bounds only when less\n"
            "                              than 5%% of the sample passes them) or FFT-based distances,\n"
            "                              whichever the calibrated costs predict to be cheaper.\n"
            "  --scan-fallback-fraction F  Adaptive: do not search the index when more than this\n"
            "                              fraction of a sample of the rank's windows pass the lower\n"
            "                              bound (default: 0.01; negative: never search the index).\n"
            "  --output PATH               KNN CSV path or threshold output prefix.\n"
            "                              Defaults: tremor_knn.csv or tremor_threshold.\n"
            "\n"
            "Benchmarking:\n"
            "  --benchmark-output PATH     Per-rank CSV with columns\n"
            "                              node_rank,index_time,query_time (seconds),index_gb.\n"
            "                              Default: <--output>.timings.csv.\n"
            "  --no-benchmark-report       Skip writing the benchmark CSV report.\n"
            "\n"
            "TREMOR parameters:\n"
            "  --search-workers N          Search/query worker threads (default: 4).\n"
            "  --index-threads N           Index construction threads (default: 4).\n"
            "  --query-threads N           Alias of --search-workers (deprecated).\n"
            "  --leaf-size N               iSAX leaf size (default: 2000).\n"
            "  --paa-segments N            PAA segment count (default: 16).\n"
            "  --replication-groups N      MPI replication groups, 0 means one per rank (default: 0).\n"
            "  --pq-threshold-divisor N    Priority-queue threshold divisor (default: 16).\n"
            "  --merge-offset N            Merge nearby offsets within this gap (default: 0).\n"
            "  --verbose                   Enable verbose TREMOR logging.\n"
            "\n"
            "Reporting:\n"
            "  --print-knn-results         Print a small KNN preview on rank 0.\n"
            "  --print-query-limit N       Number of queries to print with --print-knn-results (default: 5).\n"
            "  --help                      Show this message.\n"
            "\n"
            "Notes:\n"
            "  Files must contain little-endian float32 data on this platform.\n"
            "  --window-length must be divisible by 8 and by --paa-segments.\n",
            program);
    }

    const char *mode_name(SearchMode mode)
    {
        return mode == SearchMode::KNN ? "knn" : "threshold";
    }

    const char *refinement_name(RefinementMode mode)
    {
        return mode == RefinementMode::Leaf ? "leaf" : "adaptive";
    }

    RefinementMode parse_refinement(const std::string &value)
    {
        if (value == "leaf")
            return RefinementMode::Leaf;
        if (value == "adaptive")
            return RefinementMode::Adaptive;
        throw std::invalid_argument("Unknown --refinement: " + value + " (expected leaf or adaptive)");
    }

    std::string require_value(int argc, char **argv, int &i, const std::string &option)
    {
        if (i + 1 >= argc)
        {
            throw std::invalid_argument(option + " requires a value");
        }
        ++i;
        return argv[i];
    }

    idx_t parse_idx_t(const std::string &value, const std::string &option)
    {
        if (value.empty() || value[0] == '-')
        {
            throw std::invalid_argument("Invalid integer for " + option + ": " + value);
        }

        size_t parsed = 0;
        unsigned long long result = 0;
        try
        {
            result = std::stoull(value, &parsed, 10);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument("Invalid integer for " + option + ": " + value);
        }
        if (parsed != value.size())
        {
            throw std::invalid_argument("Invalid integer for " + option + ": " + value);
        }
        if (result > std::numeric_limits<idx_t>::max())
        {
            throw std::invalid_argument(option + " is too large for this TREMOR build");
        }
        return static_cast<idx_t>(result);
    }

    int parse_int(const std::string &value, const std::string &option)
    {
        const idx_t parsed = parse_idx_t(value, option);
        if (parsed > static_cast<idx_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument(option + " is too large for this driver");
        }
        return static_cast<int>(parsed);
    }

    float parse_float(const std::string &value, const std::string &option)
    {
        size_t parsed = 0;
        float result = 0.0f;
        try
        {
            result = std::stof(value, &parsed);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument("Invalid float for " + option + ": " + value);
        }
        if (parsed != value.size())
        {
            throw std::invalid_argument("Invalid float for " + option + ": " + value);
        }
        if (!std::isfinite(result))
        {
            throw std::invalid_argument("Invalid finite float for " + option + ": " + value);
        }
        return result;
    }

    SearchMode parse_mode(const std::string &value)
    {
        if (value == "knn" || value == "KNN")
        {
            return SearchMode::KNN;
        }
        if (value == "threshold" || value == "THRESHOLD")
        {
            return SearchMode::Threshold;
        }
        throw std::invalid_argument("Unknown --mode: " + value);
    }

    DriverOptions parse_args(int argc, char **argv)
    {
        DriverOptions options;

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];

            if (arg == "--help" || arg == "-h")
            {
                throw HelpRequested{};
            }
            else if (arg == "--dataset")
            {
                options.dataset_path = require_value(argc, argv, i, arg);
            }
            else if (arg == "--queries")
            {
                options.queries_path = require_value(argc, argv, i, arg);
            }
            else if (arg == "--output")
            {
                options.output_path = require_value(argc, argv, i, arg);
            }
            else if (arg == "--benchmark-output")
            {
                options.benchmark_output_path = require_value(argc, argv, i, arg);
            }
            else if (arg == "--no-benchmark-report")
            {
                options.skip_benchmark_report = true;
            }
            else if (arg == "--dataset-size")
            {
                options.dataset_size = parse_idx_t(require_value(argc, argv, i, arg), arg);
                options.dataset_size_set = true;
            }
            else if (arg == "--query-count")
            {
                options.query_count = parse_idx_t(require_value(argc, argv, i, arg), arg);
                options.query_count_set = true;
            }
            else if (arg == "--window-length")
            {
                options.window_length = parse_idx_t(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--mode")
            {
                options.mode = parse_mode(require_value(argc, argv, i, arg));
            }
            else if (arg == "--method")
            {
                const std::string method = require_value(argc, argv, i, arg);
                if (method != "tremor" && method != "skip-sequential" && method != "sequential")
                    throw std::invalid_argument("Unknown --method: " + method +
                                                " (expected tremor, skip-sequential, or sequential)");
                options.skip_sequential = method != "tremor";
                options.lower_bounds = method != "sequential";
            }
            else if (arg == "--scan-fallback-fraction")
            {
                // A negative fraction never searches the index (experiments: TREMOR without its index).
                options.scan_fallback_fraction = std::stod(require_value(argc, argv, i, arg));
                if (std::isnan(options.scan_fallback_fraction))
                    throw std::invalid_argument("--scan-fallback-fraction must be a number");
            }
            else if (arg == "--refinement")
            {
                options.refinement = parse_refinement(require_value(argc, argv, i, arg));
            }
            else if (arg == "--k")
            {
                options.knn_k = parse_idx_t(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--threshold")
            {
                options.threshold_correlation = parse_float(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--threshold-result-slots")
            {
                options.threshold_result_slots = parse_idx_t(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--search-workers")
            {
                options.search_workers = parse_int(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--index-threads")
            {
                options.index_threads = parse_int(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--query-threads")
            {
                // Backward-compatible alias: query and search workers are unified.
                options.search_workers = parse_int(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--leaf-size")
            {
                options.leaf_size = parse_int(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--paa-segments")
            {
                options.paa_segments = parse_int(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--replication-groups")
            {
                options.replication_groups = parse_int(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--pq-threshold-divisor")
            {
                options.pq_threshold_divisor = parse_int(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--merge-offset")
            {
                options.merge_offset = parse_int(require_value(argc, argv, i, arg), arg);
            }
            else if (arg == "--verbose")
            {
                options.verbose = true;
            }
            else if (arg == "--print-knn-results")
            {
                options.print_knn_results = true;
            }
            else if (arg == "--print-query-limit")
            {
                options.print_query_limit = parse_idx_t(require_value(argc, argv, i, arg), arg);
            }
            else
            {
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        if (options.dataset_path.empty())
        {
            throw std::invalid_argument("--dataset is required");
        }
        if (options.queries_path.empty())
        {
            throw std::invalid_argument("--queries is required");
        }
        if (options.window_length == 0)
        {
            throw std::invalid_argument("--window-length must be greater than 0");
        }
        if (options.window_length % 8 != 0)
        {
            throw std::invalid_argument("--window-length must be divisible by 8");
        }
        if (options.paa_segments <= 0)
        {
            throw std::invalid_argument("--paa-segments must be greater than 0");
        }
        if (options.window_length % static_cast<idx_t>(options.paa_segments) != 0)
        {
            throw std::invalid_argument("--window-length must be divisible by --paa-segments");
        }
        if (options.dataset_size_set && options.dataset_size < options.window_length)
        {
            throw std::invalid_argument("--dataset-size must be at least --window-length");
        }
        if (options.query_count_set && options.query_count == 0)
        {
            throw std::invalid_argument("--query-count must be greater than 0");
        }
        if (options.knn_k == 0)
        {
            throw std::invalid_argument("--k must be greater than 0");
        }
        if (options.threshold_result_slots == 0)
        {
            throw std::invalid_argument("--threshold-result-slots must be greater than 0");
        }
        if (options.threshold_correlation < -1.0f || options.threshold_correlation > 1.0f)
        {
            throw std::invalid_argument("--threshold must be between -1.0 and 1.0");
        }
        if (options.search_workers <= 0 || options.index_threads <= 0)
        {
            throw std::invalid_argument("thread counts must be greater than 0");
        }
        if (options.leaf_size <= 0)
        {
            throw std::invalid_argument("--leaf-size must be greater than 0");
        }
        if (options.pq_threshold_divisor <= 0)
        {
            throw std::invalid_argument("--pq-threshold-divisor must be greater than 0");
        }
        if (options.merge_offset < 0)
        {
            throw std::invalid_argument("--merge-offset must be non-negative");
        }
        if (options.replication_groups < 0)
        {
            throw std::invalid_argument("--replication-groups must be non-negative");
        }
        if (options.window_length > static_cast<idx_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument("--window-length is too large for this TREMOR build");
        }
        if (options.query_count_set &&
            options.query_count > static_cast<idx_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument("--query-count is too large for this TREMOR build");
        }
        if (options.knn_k > static_cast<idx_t>(std::numeric_limits<int>::max()) ||
            options.threshold_result_slots > static_cast<idx_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument("requested result count is too large for this TREMOR build");
        }

        if (options.output_path.empty())
        {
            options.output_path =
                options.mode == SearchMode::KNN ? "tremor_knn.csv" : "tremor_threshold";
        }

        if (options.benchmark_output_path.empty() && !options.skip_benchmark_report)
        {
            options.benchmark_output_path = options.output_path + ".timings.csv";
        }

        return options;
    }

    TremorConfig make_tremor_config(const DriverOptions &options)
    {
        TremorConfig config;
        config.search_workers = options.search_workers;
        config.index_threads = options.index_threads;
        config.query_threads = options.search_workers;
        config.leaf_size = options.leaf_size;
        config.paa_segments = options.paa_segments;
        config.replication_groups = options.replication_groups;
        config.pq_th_div_factor = options.pq_threshold_divisor;
        return config;
    }

    idx_t checked_product(idx_t lhs, idx_t rhs, const std::string &label)
    {
        if (lhs != 0 && rhs > std::numeric_limits<idx_t>::max() / lhs)
        {
            throw std::overflow_error(label + " is too large");
        }
        return lhs * rhs;
    }

    idx_t file_float_count(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error("Cannot open " + path + ": " + std::strerror(errno));
        }

        const std::streamoff bytes = file.tellg();
        if (bytes < 0)
        {
            throw std::runtime_error("Cannot determine file size for " + path);
        }
        if (bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        {
            throw std::runtime_error(path + " size is not a multiple of sizeof(float)");
        }

        return static_cast<idx_t>(bytes / static_cast<std::streamoff>(sizeof(float)));
    }

    std::vector<float> read_float_file(const std::string &path, idx_t count, const std::string &label)
    {
        if (count == 0)
        {
            throw std::invalid_argument(label + " count must be greater than 0");
        }
        if (count > static_cast<idx_t>(std::numeric_limits<size_t>::max() / sizeof(float)))
        {
            throw std::overflow_error(label + " is too large for this process");
        }

        std::vector<float> values(static_cast<size_t>(count));
        const size_t byte_count = values.size() * sizeof(float);
        if (byte_count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
        {
            throw std::overflow_error(label + " is too large for one read");
        }

        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Cannot open " + path + ": " + std::strerror(errno));
        }

        file.read(reinterpret_cast<char *>(values.data()),
                  static_cast<std::streamsize>(byte_count));
        if (!file)
        {
            throw std::runtime_error("Failed to read " + label + " from " + path);
        }

        return values;
    }

    struct LoadedInputs
    {
        std::vector<float> queries;
        idx_t dataset_size = 0;
        idx_t query_count = 0;
    };

    LoadedInputs load_inputs(const DriverOptions &options)
    {
        LoadedInputs inputs;

        const idx_t dataset_file_floats = file_float_count(options.dataset_path);
        inputs.dataset_size = options.dataset_size_set ? options.dataset_size : dataset_file_floats;
        if (inputs.dataset_size > dataset_file_floats)
        {
            throw std::runtime_error("--dataset-size exceeds float count in " + options.dataset_path);
        }
        if (inputs.dataset_size < options.window_length)
        {
            throw std::runtime_error("dataset has fewer samples than --window-length");
        }

        const idx_t query_file_floats = file_float_count(options.queries_path);
        if (options.query_count_set)
        {
            inputs.query_count = options.query_count;
            const idx_t required_query_floats =
                checked_product(inputs.query_count, options.window_length, "query data");
            if (required_query_floats > query_file_floats)
            {
                throw std::runtime_error("--query-count * --window-length exceeds float count in " +
                                         options.queries_path);
            }
        }
        else
        {
            if (query_file_floats % options.window_length != 0)
            {
                throw std::runtime_error("query file float count is not divisible by --window-length");
            }
            inputs.query_count = query_file_floats / options.window_length;
            if (inputs.query_count == 0)
            {
                throw std::runtime_error("query file contains no complete query vectors");
            }
            if (inputs.query_count > static_cast<idx_t>(std::numeric_limits<int>::max()))
            {
                throw std::runtime_error("inferred query count is too large for this TREMOR build");
            }
        }

        inputs.queries = read_float_file(
            options.queries_path,
            checked_product(inputs.query_count, options.window_length, "query data"),
            "queries");

        return inputs;
    }

    float squared_ed_to_correlation(float squared_distance, idx_t window_length)
    {
        return 1.0f - squared_distance / (2.0f * static_cast<float>(window_length));
    }

    void print_options(const DriverOptions &options, const LoadedInputs &inputs, int world_size)
    {
        std::printf("\nTREMOR external-data run\n");
        std::printf("  MPI ranks: %d\n", world_size);
        std::printf("  mode: %s\n", mode_name(options.mode));
        std::printf("  dataset: %s\n", options.dataset_path.c_str());
        std::printf("  dataset_size: %llu float32 samples\n",
                    static_cast<unsigned long long>(inputs.dataset_size));
        std::printf("  queries: %s\n", options.queries_path.c_str());
        std::printf("  query_count: %llu\n", static_cast<unsigned long long>(inputs.query_count));
        std::printf("  window_length: %llu\n",
                    static_cast<unsigned long long>(options.window_length));
        std::printf("  output: %s\n", options.output_path.c_str());

        if (options.mode == SearchMode::KNN)
        {
            std::printf("  k: %llu\n", static_cast<unsigned long long>(options.knn_k));
        }
        else
        {
            std::printf("  threshold_correlation: %0.6f\n", options.threshold_correlation);
            std::printf("  threshold_result_slots: %llu\n",
                        static_cast<unsigned long long>(options.threshold_result_slots));
        }

        std::printf("  search_workers: %d\n", options.search_workers);
        std::printf("  query_threads: %d (alias of search_workers)\n", options.search_workers);
        std::printf("  index_threads: %d\n", options.index_threads);
        std::printf("  leaf_size: %d\n", options.leaf_size);
        std::printf("  paa_segments: %d\n", options.paa_segments);
        std::printf("  replication_groups: %d\n", options.replication_groups);
        std::printf("  pq_threshold_divisor: %d\n", options.pq_threshold_divisor);
        std::printf("  merge_offset: %d\n", options.merge_offset);
        std::printf("  method: %s\n", !options.skip_sequential ? "tremor" : options.lower_bounds ? "skip-sequential"
                                                                                                 : "sequential");
        std::printf("  refinement: %s\n", refinement_name(options.refinement));
        if (options.refinement == RefinementMode::Adaptive)
            std::printf("  scan_fallback_fraction: %g\n", options.scan_fallback_fraction);
        std::printf("  verbose: %s\n", options.verbose ? "true" : "false");
    }

    void print_knn_preview(const std::vector<idx_t> &indices,
                           const std::vector<float> &distances,
                           idx_t query_count,
                           idx_t nearest_neighbors,
                           idx_t window_length,
                           idx_t query_limit)
    {
        const idx_t queries_to_print = query_count < query_limit ? query_count : query_limit;
        std::printf("\nKNN search preview\n");
        for (idx_t q = 0; q < queries_to_print; ++q)
        {
            std::printf("  query %llu\n", static_cast<unsigned long long>(q));
            for (idx_t k = 0; k < nearest_neighbors; ++k)
            {
                const size_t result_offset = static_cast<size_t>(q * nearest_neighbors + k);
                const float corr = squared_ed_to_correlation(distances[result_offset], window_length);
                std::printf("    #%llu: offset=%llu squared_ed=%0.6f corr=%0.6f\n",
                            static_cast<unsigned long long>(k + 1),
                            static_cast<unsigned long long>(indices[result_offset]),
                            distances[result_offset],
                            corr);
            }
        }
    }

    void finalize_mpi_if_needed()
    {
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);

        if (initialized && !finalized)
        {
            MPI_Finalize();
        }
    }

    void write_timings_csv(const std::string &path,
                           const std::vector<double> &index_times,
                           const std::vector<double> &query_times,
                           const std::vector<double> &index_gb)
    {
        if (path.empty())
        {
            return;
        }

        std::FILE *out = std::fopen(path.c_str(), "w");
        if (out == nullptr)
        {
            std::fprintf(stderr,
                         "Warning: cannot open benchmark report file %s: %s\n",
                         path.c_str(),
                         std::strerror(errno));
            return;
        }

        std::fprintf(out, "node_rank,index_time,query_time,index_gb\n");
        const size_t rank_count = index_times.size();
        for (size_t rank = 0; rank < rank_count; ++rank)
        {
            std::fprintf(out,
                         "%zu,%0.6f,%0.6f,%0.3f\n",
                         rank,
                         index_times[rank],
                         query_times[rank],
                         index_gb[rank]);
        }

        std::fclose(out);
    }
}

namespace
{
    // Gathers every rank's timings and index memory; rank 0 writes the benchmark CSV.
    void report_timings(const DriverOptions &options, int rank, int world_size, const BenchmarkTimings &timings,
                        double index_gb)
    {
        std::vector<double> all_index_times(rank == 0 ? world_size : 0);
        std::vector<double> all_search_times(rank == 0 ? world_size : 0);
        std::vector<double> all_index_gb(rank == 0 ? world_size : 0);
        MPI_Gather(&timings.index_build_seconds, 1, MPI_DOUBLE, all_index_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&timings.search_seconds, 1, MPI_DOUBLE, all_search_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&index_gb, 1, MPI_DOUBLE, all_index_gb.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0 && !options.skip_benchmark_report)
        {
            write_timings_csv(options.benchmark_output_path, all_index_times, all_search_times, all_index_gb);
            std::printf("Benchmark timings CSV: %s\n", options.benchmark_output_path.c_str());
        }
    }

    // --method skip-sequential or sequential: same inputs, outputs and timings as TREMOR, without an index tree.
    void run_skip_sequential(const DriverOptions &options, int argc, char **argv)
    {
        int initialized = 0, provided = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
            MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        int rank = 0, world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        const double query_input_start = MPI_Wtime();
        LoadedInputs inputs = load_inputs(options);
        const double query_input_seconds = MPI_Wtime() - query_input_start;
        if (rank == 0)
            print_options(options, inputs, world_size);

        SkipSequentialScan scan(options.search_workers, options.paa_segments, options.merge_offset,
                                options.replication_groups, options.lower_bounds);
        BenchmarkTimings timings;
        MPI_Barrier(MPI_COMM_WORLD);
        const double build_start = MPI_Wtime();
        scan.build(options.dataset_path, inputs.dataset_size, static_cast<int>(options.window_length));
        timings.index_build_seconds = MPI_Wtime() - build_start;

        MPI_Barrier(MPI_COMM_WORLD);
        const double search_start = MPI_Wtime();
        if (options.mode == SearchMode::KNN)
            scan.search_knn(inputs.queries.data(), static_cast<int>(inputs.query_count),
                            static_cast<int>(options.knn_k), options.output_path);
        else
            scan.search_threshold(inputs.queries.data(), static_cast<int>(inputs.query_count),
                                  options.threshold_correlation, options.output_path);
        timings.search_seconds = query_input_seconds + MPI_Wtime() - search_start;

        report_timings(options, rank, world_size, timings, scan.summary_memory_bytes() / 1e9);
    }
}

int main(int argc, char **argv)
{
    int exit_code = EXIT_SUCCESS;

    try
    {
        const DriverOptions options = parse_args(argc, argv);
        const TremorConfig config = make_tremor_config(options);

        if (options.skip_sequential)
        {
            run_skip_sequential(options, argc, argv);
        }
        else
        {
            Tremor tremor(config, DistanceType::L2_SQUARED, argc, argv);

            tremor.setNumThreads(options.search_workers);
            tremor.setIndexThreads(options.index_threads);
            tremor.setQueryThreads(options.search_workers);
            tremor.setReplicationGroups(options.replication_groups);
            tremor.setVerbose(options.verbose);
            tremor.setMergeOffset(options.merge_offset);
            tremor.setRefinementMode(options.refinement);
            set_scan_fallback_fraction(options.scan_fallback_fraction);

            const int rank = tremor.getMyRank();
            const int world_size = tremor.getCommSz();
            if (options.replication_groups > world_size)
            {
                throw std::invalid_argument("--replication-groups cannot exceed MPI ranks");
            }

            const double query_input_start = MPI_Wtime();
            LoadedInputs inputs = load_inputs(options);
            const double query_input_seconds = MPI_Wtime() - query_input_start;
            const idx_t available_subsequences = inputs.dataset_size - options.window_length + static_cast<idx_t>(1);
            if (options.mode == SearchMode::KNN && options.knn_k > available_subsequences)
            {
                throw std::invalid_argument("--k cannot exceed the number of indexed subsequences");
            }

            if (rank == 0)
            {
                print_options(options, inputs, world_size);
            }

            BenchmarkTimings timings;

            MPI_Barrier(MPI_COMM_WORLD);
            const double build_start = MPI_Wtime();
            tremor.buildIndexLongSequence(
                options.dataset_path,
                inputs.dataset_size,
                options.window_length);
            timings.index_build_seconds = MPI_Wtime() - build_start;

            tremor.setOutputFile(options.output_path);

            if (options.mode == SearchMode::KNN)
            {
                tremor.setModeKNN();

                std::vector<idx_t> indices(static_cast<size_t>(checked_product(inputs.query_count, options.knn_k, "KNN results")));
                std::vector<float> distances(indices.size());

                MPI_Barrier(MPI_COMM_WORLD);
                const double search_start = MPI_Wtime();
                tremor.searchIndex(
                    inputs.queries.data(),
                    inputs.query_count,
                    options.knn_k,
                    indices.data(),
                    distances.data());
                timings.search_seconds = query_input_seconds + MPI_Wtime() - search_start;

                if (rank == 0)
                {
                    std::printf("\nKNN search complete\n");
                    std::printf("KNN CSV output: %s\n", options.output_path.c_str());
                    if (options.print_knn_results)
                    {
                        print_knn_preview(
                            indices,
                            distances,
                            inputs.query_count,
                            options.knn_k,
                            options.window_length,
                            options.print_query_limit);
                    }
                }
            }
            else
            {
                tremor.setModeThreshold();
                tremor.setCorrelationThreshold(options.threshold_correlation);

                std::vector<idx_t> indices(static_cast<size_t>(checked_product(inputs.query_count, options.threshold_result_slots, "threshold results")));
                std::vector<float> distances(indices.size());

                MPI_Barrier(MPI_COMM_WORLD);
                const double search_start = MPI_Wtime();
                tremor.searchIndex(
                    inputs.queries.data(),
                    inputs.query_count,
                    options.threshold_result_slots,
                    indices.data(),
                    distances.data());
                timings.search_seconds = query_input_seconds + MPI_Wtime() - search_start;

                if (rank == 0)
                {
                    std::printf("\nThreshold search complete at correlation >= %0.6f\n", options.threshold_correlation);
                    std::printf("Threshold CSV output prefix: %s_<rank>.csv\n", options.output_path.c_str());
                }
            }

            if (options.refinement == RefinementMode::Adaptive)
            {
                long long leaf_queries = 0, scan_queries = 0;
                adaptive_refinement_counts(&leaf_queries, &scan_queries);
                std::printf("[Node %d] adaptive refinement: %lld queries leaf order, %lld queries scan\n", rank, leaf_queries, scan_queries);
            }

            report_timings(options, rank, world_size, timings, tremor.getIndexMemoryBytes() / 1e9);
        }
    }
    catch (const HelpRequested &)
    {
        print_usage(argv[0]);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "TREMOR driver failed: %s\n", ex.what());
        std::fprintf(stderr, "Run with --help for usage.\n");
        exit_code = EXIT_FAILURE;
    }

    finalize_mpi_if_needed();
    return exit_code;
}
