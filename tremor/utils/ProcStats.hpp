#pragma once
// Diagnostic build only (-DTREMOR_STATS): kernel memory counters (NUMA balancing, page migration, transparent huge
// pages), the memory policy of the large mappings, and the CPU time of this process's threads.
#ifdef TREMOR_STATS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <unistd.h>

inline void tremor_print_vmstat(int rank, const char *tag)
{
    FILE *f = std::fopen("/proc/vmstat", "r");
    if (f == nullptr)
        return;
    const char *keys[] = {"numa_hint_faults ", "numa_hint_faults_local ", "numa_pages_migrated ", "pgmigrate_success ",
                          "thp_fault_alloc ", "thp_collapse_alloc ", "numa_pte_updates "};
    char line[256];
    std::printf("VMSTAT rank=%d tag=%s", rank, tag);
    while (std::fgets(line, sizeof(line), f))
        for (const char *key : keys)
            if (std::strncmp(line, key, std::strlen(key)) == 0)
                std::printf(" %.*s=%s", static_cast<int>(std::strlen(key) - 1), key, line + std::strlen(key));
    std::fclose(f);
    std::printf("\n");
}

inline void tremor_print_proc_summary(int rank)
{
    // Memory policy of the mappings above 1 GB.
    if (FILE *f = std::fopen("/proc/self/numa_maps", "r"))
    {
        char line[4096];
        while (std::fgets(line, sizeof(line), f))
        {
            const char *anon = std::strstr(line, "anon=");
            const long pages = anon ? std::atol(anon + 5) : 0;
            const char *huge = std::strstr(line, "kernelpagesize_kB=2048");
            if (pages * (huge ? 2048L : 4L) > 1024L * 1024L) // above 1 GB
            {
                line[std::strcspn(line, "\n")] = 0;
                std::printf("NUMAMAPS rank=%d %.160s\n", rank, line);
            }
        }
        std::fclose(f);
    }
    // Threads that used more than one second of CPU.
    const long ticks = sysconf(_SC_CLK_TCK);
    if (DIR *dir = opendir("/proc/self/task"))
    {
        int threads = 0;
        double total = 0.0;
        while (dirent *entry = readdir(dir))
        {
            if (entry->d_name[0] == '.')
                continue;
            char path[256], buffer[1024];
            std::snprintf(path, sizeof(path), "/proc/self/task/%s/stat", entry->d_name);
            FILE *f = std::fopen(path, "r");
            if (f == nullptr)
                continue;
            const size_t n = std::fread(buffer, 1, sizeof(buffer) - 1, f);
            std::fclose(f);
            buffer[n] = 0;
            const char *close = std::strrchr(buffer, ')'); // after the command name
            if (close == nullptr)
                continue;
            unsigned long utime = 0, stime = 0;
            // fields after ")": state(3) ... utime(14) stime(15)
            if (std::sscanf(close + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &utime, &stime) != 2)
                continue;
            const double seconds = static_cast<double>(utime + stime) / static_cast<double>(ticks);
            threads++;
            total += seconds;
            if (seconds > 1.0)
            {
                char name[64] = "?";
                const char *open = std::strchr(buffer, '(');
                if (open != nullptr)
                    std::snprintf(name, sizeof(name), "%.*s", static_cast<int>(close - open - 1), open + 1);
                std::printf("THREADCPU rank=%d tid=%s name=%s cpu_s=%.1f\n", rank, entry->d_name, name, seconds);
            }
        }
        closedir(dir);
        std::printf("THREADCPU rank=%d threads=%d total_cpu_s=%.1f\n", rank, threads, total);
    }
}
// CPUs and NUMA nodes this process may use, and the NUMA policy and placement of the mapping that holds `ptr`.
inline void tremor_print_placement(int rank, const char *name, const void *ptr)
{
    if (FILE *f = std::fopen("/proc/self/status", "r"))
    {
        char line[4096];
        while (std::fgets(line, sizeof(line), f))
            if (std::strncmp(line, "Cpus_allowed_list:", 18) == 0 || std::strncmp(line, "Mems_allowed_list:", 18) == 0)
            {
                line[std::strcspn(line, "\n")] = 0;
                std::printf("PLACEMENT rank=%d %s\n", rank, line);
            }
        std::fclose(f);
    }
    if (FILE *f = std::fopen("/proc/self/numa_maps", "r"))
    {
        const unsigned long address = reinterpret_cast<unsigned long>(ptr);
        unsigned long best = 0;
        char line[4096], found[4096] = "";
        while (std::fgets(line, sizeof(line), f))
        {
            unsigned long start = std::strtoul(line, nullptr, 16);
            if (start <= address && start >= best)
            {
                best = start;
                std::snprintf(found, sizeof(found), "%s", line);
            }
        }
        std::fclose(f);
        found[std::strcspn(found, "\n")] = 0;
        std::printf("PLACEMENT rank=%d array=%s %.200s\n", rank, name, found);
    }
}
#endif
