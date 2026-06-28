#include <cmath>
#include <vector>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <omp.h>

// SIMD width (doubles per vector): 8 on AVX-512 hardware, otherwise 4.
#if defined(__AVX512F__)
static constexpr int VW = 8;
#else
static constexpr int VW = 4;
#endif

typedef double dvec __attribute__ ((vector_size (VW * sizeof(double))));

// Register-blocking micro-kernel dimensions (MR rows x NR cols).
// NR == 2 * VW == two SIMD vectors of doubles.
static constexpr int MR = 6;
static constexpr int NR = 2 * VW;

// Smallest row count divisible by both MR and NR (so every micro-tile is full).
static constexpr int PAD = (VW == 8) ? 48 : 24;

// Cache-blocking dimensions.
//   KC : panel depth (k dimension)
//   MC : rows of A kept resident in L2 (multiple of MR)
//   NC : cols of B kept resident in L3 (multiple of NR and MC)
static constexpr int KC = 256;
static constexpr int MC = 72;
static constexpr int NC = 4032;

static inline dvec bcast(double x) {
    dvec v;
    for (int i = 0; i < VW; i++) v[i] = x;
    return v;
}

static inline dvec loadu(const double *p) {
    dvec v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static inline void storeu(double *p, dvec v) {
    std::memcpy(p, &v, sizeof(v));
}

// Pack a panel of "count" consecutive rows of X (row-major, leading dim ldX)
// over the k-range [k0, k0+kk) into "dst" with layout dst[k*count + r].
// "count" is MR (for A) or NR (for B).
static inline void pack_panel(double *dst, const double *X, int ldX,
                              int row0, int count, int k0, int kk) {
    for (int r = 0; r < count; r++) {
        const double *src = X + static_cast<size_t>(row0 + r) * ldX + k0;
        for (int k = 0; k < kk; k++) {
            dst[k * count + r] = src[k];
        }
    }
}

// Pack an MC-block of A (rows [i0, i0+mc)) into Ap, panel by panel of MR rows.
static void pack_A(double *Ap, const double *X, int ldX,
                   int i0, int mc, int k0, int kk) {
    double *dst = Ap;
    for (int ir = 0; ir < mc; ir += MR) {
        pack_panel(dst, X, ldX, i0 + ir, MR, k0, kk);
        dst += static_cast<size_t>(MR) * kk;
    }
}

// Pack an NC-block of B (== columns j of X^T == rows j of X) into Bp,
// panel by panel of NR columns. Panels are independent, so parallelise.
static void pack_B(double *Bp, const double *X, int ldX,
                   int j0, int nc, int k0, int kk) {
    #pragma omp parallel for schedule(static)
    for (int jr = 0; jr < nc; jr += NR) {
        double *dst = Bp + static_cast<size_t>(jr / NR) * NR * kk;
        pack_panel(dst, X, ldX, j0 + jr, NR, k0, kk);
    }
}

// Micro-kernel: C[0..MR, 0..NR] += Ap_panel * Bp_panel  (depth kk).
// Ap_panel[k*MR + r], Bp_panel[k*NR + c]. C has leading dimension ldc.
static inline void micro_kernel(const double *Ap, const double *Bp,
                                double *C, int ldc, int kk) {
    const dvec dzero = bcast(0.0);
    dvec r00 = dzero, r01 = dzero;
    dvec r10 = dzero, r11 = dzero;
    dvec r20 = dzero, r21 = dzero;
    dvec r30 = dzero, r31 = dzero;
    dvec r40 = dzero, r41 = dzero;
    dvec r50 = dzero, r51 = dzero;

    for (int k = 0; k < kk; k++) {
        dvec b0 = loadu(Bp + k * NR + 0);
        dvec b1 = loadu(Bp + k * NR + VW);
        const double *a = Ap + k * MR;
        // GCC vector extensions broadcast the scalar across the vector.
        r00 += a[0] * b0; r01 += a[0] * b1;
        r10 += a[1] * b0; r11 += a[1] * b1;
        r20 += a[2] * b0; r21 += a[2] * b1;
        r30 += a[3] * b0; r31 += a[3] * b1;
        r40 += a[4] * b0; r41 += a[4] * b1;
        r50 += a[5] * b0; r51 += a[5] * b1;
    }

    double *c0 = C + 0 * ldc;
    double *c1 = C + 1 * ldc;
    double *c2 = C + 2 * ldc;
    double *c3 = C + 3 * ldc;
    double *c4 = C + 4 * ldc;
    double *c5 = C + 5 * ldc;
    storeu(c0 + 0, loadu(c0 + 0) + r00); storeu(c0 + VW, loadu(c0 + VW) + r01);
    storeu(c1 + 0, loadu(c1 + 0) + r10); storeu(c1 + VW, loadu(c1 + VW) + r11);
    storeu(c2 + 0, loadu(c2 + 0) + r20); storeu(c2 + VW, loadu(c2 + VW) + r21);
    storeu(c3 + 0, loadu(c3 + 0) + r30); storeu(c3 + VW, loadu(c3 + VW) + r31);
    storeu(c4 + 0, loadu(c4 + 0) + r40); storeu(c4 + VW, loadu(c4 + VW) + r41);
    storeu(c5 + 0, loadu(c5 + 0) + r50); storeu(c5 + VW, loadu(c5 + VW) + r51);
}

void correlate(int ny, int nx, const float *data, float *result) {
    if (ny <= 0) return;

    const int K = nx;
    // Pad the row count up to a multiple of lcm(MR, NR) so that every
    // micro-tile is full (padded rows are zero and contribute nothing).
    const int Mp = ((ny + PAD - 1) / PAD) * PAD;

    std::vector<double> X(static_cast<size_t>(Mp) * K, 0.0);

    // Normalise each real row to zero mean and unit length.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; i++) {
        const float *row = &data[static_cast<size_t>(i) * nx];
        double mean = 0.0;
        for (int x = 0; x < nx; x++) mean += row[x];
        mean /= static_cast<double>(nx);

        double ss = 0.0;
        for (int x = 0; x < nx; x++) {
            double v = static_cast<double>(row[x]) - mean;
            ss += v * v;
        }
        const double inv = (ss > 0.0) ? 1.0 / std::sqrt(ss) : 0.0;

        double *xr = &X[static_cast<size_t>(i) * K];
        for (int x = 0; x < nx; x++) {
            xr[x] = (static_cast<double>(row[x]) - mean) * inv;
        }
    }

    // C = X * X^T (only the lower triangle i >= j is needed), accumulated in
    // double precision, then written to the float result.
    std::vector<double> C(static_cast<size_t>(Mp) * Mp, 0.0);

    const int nthreads = std::max(1, omp_get_max_threads());
    std::vector<double> Apbuf(static_cast<size_t>(nthreads) * MC * KC);
    std::vector<double> Bp(static_cast<size_t>(KC) * NC);

    for (int jc = 0; jc < Mp; jc += NC) {
        const int nc = std::min(NC, Mp - jc);
        for (int kc = 0; kc < K; kc += KC) {
            const int kk = std::min(KC, K - kc);
            pack_B(Bp.data(), X.data(), K, jc, nc, kc, kk);

            // jc is a multiple of MC, so all rows below jc are entirely above
            // the diagonal for these columns and can be skipped.
            #pragma omp parallel for schedule(dynamic, 1)
            for (int ic = jc; ic < Mp; ic += MC) {
                const int mc = std::min(MC, Mp - ic);
                double *Ap = &Apbuf[static_cast<size_t>(omp_get_thread_num()) * MC * KC];
                pack_A(Ap, X.data(), K, ic, mc, kc, kk);

                for (int jr = 0; jr < nc; jr += NR) {
                    const double *Bpanel =
                        Bp.data() + static_cast<size_t>(jr / NR) * NR * kk;
                    const int jcol = jc + jr;
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int irow = ic + ir;
                        // Skip micro-tiles that lie fully above the diagonal.
                        if (irow + MR - 1 < jcol) continue;
                        const double *Apanel =
                            Ap + static_cast<size_t>(ir / MR) * MR * kk;
                        double *Ctile = &C[static_cast<size_t>(irow) * Mp + jcol];
                        micro_kernel(Apanel, Bpanel, Ctile, Mp, kk);
                    }
                }
            }
        }
    }

    // Write the lower triangle (i >= j) into the result.
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            result[i + static_cast<size_t>(j) * ny] =
                static_cast<float>(C[static_cast<size_t>(i) * Mp + j]);
        }
    }
}
