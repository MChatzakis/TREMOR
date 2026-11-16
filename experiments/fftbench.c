// Per-template cost of the sliding dot products of D-MASS: one whole-series FFT
// (current D-MASS, MASS V2 style) vs. overlap-save over chunks of B points
// (MASS V3 style), on the same data. Prints seconds per template and the max
// difference of the dot products.
#include <fftw3.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long long next_smooth_length(long long target)
{
    long long best = LLONG_MAX;
    for (long long p2 = 1; p2 < best; p2 *= 2)
        for (long long p3 = p2; p3 < best; p3 *= 3)
            for (long long p5 = p3; p5 < best; p5 *= 5)
                for (long long p7 = p5; p7 < best; p7 *= 7)
                    if (p7 >= target) { best = p7; break; }
    return best;
}

int main(int argc, char **argv)
{
    const char *file = argv[1];
    const long n = atol(argv[2]);
    const int m = atoi(argv[3]);
    const int T = omp_get_max_threads();
    float *raw = malloc(sizeof(float) * n);
    FILE *f = fopen(file, "rb");
    if (fread(raw, sizeof(float), n, f) != (size_t)n) { printf("short read\n"); return 1; }
    fclose(f);
    double *x = fftw_alloc_real(n);
    for (long i = 0; i < n; i++) x[i] = raw[i];
    free(raw);
    const long nw = n - m + 1;
    double *y = malloc(sizeof(double) * m);
    for (int i = 0; i < m; i++) y[i] = x[123456789 % (n - m) + i];  // a subsequence as the query
    double *ref = malloc(sizeof(double) * nw), *out = malloc(sizeof(double) * nw);

    // (A) whole-series FFT, as in D-MASS.
    fftw_init_threads();
    fftw_plan_with_nthreads(T);
    const long len = next_smooth_length(n + m - 1);
    const long bins = len / 2 + 1;
    double *buf = fftw_alloc_real(len);
    fftw_complex *X = fftw_alloc_complex(bins), *Y = fftw_alloc_complex(bins);
    fftw_plan pdata = fftw_plan_dft_r2c_1d(len, buf, X, FFTW_ESTIMATE);
    fftw_plan pq = fftw_plan_dft_r2c_1d(len, buf, Y, FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_c2r_1d(len, Y, buf, FFTW_ESTIMATE);
#pragma omp parallel for
    for (long i = 0; i < len; i++) buf[i] = i < n ? x[i] : 0.0;
    fftw_execute(pdata);
    for (int rep = 0; rep < 3; rep++) {
        double t0 = omp_get_wtime(), t1, t2;
#pragma omp parallel for
        for (long i = 0; i < len; i++) buf[i] = i < m ? y[m - 1 - i] : 0.0;
        fftw_execute(pq);
        t1 = omp_get_wtime();
#pragma omp parallel for
        for (long i = 0; i < bins; i++) {
            double re = X[i][0] * Y[i][0] - X[i][1] * Y[i][1], im = X[i][1] * Y[i][0] + X[i][0] * Y[i][1];
            Y[i][0] = re; Y[i][1] = im;
        }
        fftw_execute(pb);
        t2 = omp_get_wtime();
#pragma omp parallel for
        for (long j = 0; j < nw; j++) ref[j] = buf[m - 1 + j] / len;
        printf("whole len=%ld threads=%d: %.3f s per template (query r2c %.3f, multiply+c2r %.3f)\n", len, T,
               omp_get_wtime() - t0, t1 - t0, t2 - t1);
        fflush(stdout);
    }
    fftw_destroy_plan(pdata); fftw_destroy_plan(pq); fftw_destroy_plan(pb);
    fftw_free(buf); fftw_free(X); fftw_free(Y);

    // (B) overlap-save over chunks of B points, one single-threaded FFT per chunk.
    for (int a = 4; a < argc; a++) {
        const long B = 1L << atoi(argv[a]);
        const long step = B - m + 1, nchunks = (nw + step - 1) / step, cb = B / 2 + 1;
        fftw_plan_with_nthreads(1);
        double *tb = fftw_alloc_real(B);
        fftw_complex *tc = fftw_alloc_complex(cb);
        fftw_plan fr = fftw_plan_dft_r2c_1d(B, tb, tc, FFTW_MEASURE);
        fftw_plan br = fftw_plan_dft_c2r_1d(B, tc, tb, FFTW_MEASURE);
        fftw_complex *XC = fftw_alloc_complex(nchunks * cb), *YC = fftw_alloc_complex(cb);
        double t0 = omp_get_wtime();
#pragma omp parallel
        {
            double *lb = fftw_alloc_real(B);
#pragma omp for schedule(static)
            for (long c = 0; c < nchunks; c++) {
                for (long i = 0; i < B; i++) lb[i] = c * step + i < n ? x[c * step + i] : 0.0;
                fftw_execute_dft_r2c(fr, lb, XC + c * cb);
            }
            fftw_free(lb);
        }
        double prep = omp_get_wtime() - t0;
        for (int rep = 0; rep < 3; rep++) {
            t0 = omp_get_wtime();
            for (long i = 0; i < B; i++) tb[i] = i < m ? y[m - 1 - i] : 0.0;
            fftw_execute_dft_r2c(fr, tb, YC);
#pragma omp parallel
            {
                double *lb = fftw_alloc_real(B);
                fftw_complex *lc = fftw_alloc_complex(cb);
#pragma omp for schedule(static)
                for (long c = 0; c < nchunks; c++) {
                    const fftw_complex *xc = XC + c * cb;
                    for (long i = 0; i < cb; i++) {
                        lc[i][0] = xc[i][0] * YC[i][0] - xc[i][1] * YC[i][1];
                        lc[i][1] = xc[i][1] * YC[i][0] + xc[i][0] * YC[i][1];
                    }
                    fftw_execute_dft_c2r(br, lc, lb);
                    const long end = c * step + step < nw ? c * step + step : nw;
                    for (long j = c * step; j < end; j++) out[j] = lb[m - 1 + j - c * step] / B;
                }
                fftw_free(lb); fftw_free(lc);
            }
            double t = omp_get_wtime() - t0;
            double maxdiff = 0.0;
#pragma omp parallel for reduction(max : maxdiff)
            for (long j = 0; j < nw; j++) maxdiff = fmax(maxdiff, fabs(out[j] - ref[j]));
            printf("chunks B=2^%s (%ld chunks, prep %.2f s): %.3f s per template, max |diff| %.2e\n", argv[a], nchunks,
                   prep, t, maxdiff);
            fflush(stdout);
        }
        fftw_destroy_plan(fr); fftw_destroy_plan(br);
        fftw_free(tb); fftw_free(tc); fftw_free(XC); fftw_free(YC);
    }
    return 0;
}
