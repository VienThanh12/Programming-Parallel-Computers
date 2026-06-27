/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <vector>
#include <cstddef>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

constexpr double4_t d4zero {0.0, 0.0, 0.0, 0.0};

static inline double4_t bcast(double v) {
    return (double4_t){v, v, v, v};
}

static inline double4_t loadu(const double *p) {
    return (double4_t){p[0], p[1], p[2], p[3]};
}

static inline void storeu(double *p, double4_t v) {
    p[0] = v[0]; p[1] = v[1]; p[2] = v[2]; p[3] = v[3];
}

// C(m x n) = A(m x k) * B(k x n).  Requires m % 6 == 0 and n % 8 == 0.
static void base_mul(int m, int k, int n,
                     const double *A, int lda,
                     const double *B, int ldb,
                     double *C, int ldc) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i0 = 0; i0 < m; i0 += 6) {
        const double *a0 = A + static_cast<size_t>(i0 + 0) * lda;
        const double *a1 = A + static_cast<size_t>(i0 + 1) * lda;
        const double *a2 = A + static_cast<size_t>(i0 + 2) * lda;
        const double *a3 = A + static_cast<size_t>(i0 + 3) * lda;
        const double *a4 = A + static_cast<size_t>(i0 + 4) * lda;
        const double *a5 = A + static_cast<size_t>(i0 + 5) * lda;

        for (int j0 = 0; j0 < n; j0 += 8) {
            double4_t r00 = d4zero, r01 = d4zero;
            double4_t r10 = d4zero, r11 = d4zero;
            double4_t r20 = d4zero, r21 = d4zero;
            double4_t r30 = d4zero, r31 = d4zero;
            double4_t r40 = d4zero, r41 = d4zero;
            double4_t r50 = d4zero, r51 = d4zero;

            const double *bp = B + j0;
            for (int d = 0; d < k; d++) {
                double4_t b0 = loadu(bp + 0);
                double4_t b1 = loadu(bp + 4);
                bp += ldb;
                double4_t av;
                av = bcast(a0[d]); r00 += av * b0; r01 += av * b1;
                av = bcast(a1[d]); r10 += av * b0; r11 += av * b1;
                av = bcast(a2[d]); r20 += av * b0; r21 += av * b1;
                av = bcast(a3[d]); r30 += av * b0; r31 += av * b1;
                av = bcast(a4[d]); r40 += av * b0; r41 += av * b1;
                av = bcast(a5[d]); r50 += av * b0; r51 += av * b1;
            }

            double *c0 = C + static_cast<size_t>(i0 + 0) * ldc + j0;
            double *c1 = C + static_cast<size_t>(i0 + 1) * ldc + j0;
            double *c2 = C + static_cast<size_t>(i0 + 2) * ldc + j0;
            double *c3 = C + static_cast<size_t>(i0 + 3) * ldc + j0;
            double *c4 = C + static_cast<size_t>(i0 + 4) * ldc + j0;
            double *c5 = C + static_cast<size_t>(i0 + 5) * ldc + j0;
            storeu(c0 + 0, r00); storeu(c0 + 4, r01);
            storeu(c1 + 0, r10); storeu(c1 + 4, r11);
            storeu(c2 + 0, r20); storeu(c2 + 4, r21);
            storeu(c3 + 0, r30); storeu(c3 + 4, r31);
            storeu(c4 + 0, r40); storeu(c4 + 4, r41);
            storeu(c5 + 0, r50); storeu(c5 + 4, r51);
        }
    }
}

// D(r x c) = A +/- B
static void madd(int r, int c,
                 const double *A, int lda,
                 const double *B, int ldb,
                 double sign, double *D, int ldd) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < r; i++) {
        const double *ar = A + static_cast<size_t>(i) * lda;
        const double *br = B + static_cast<size_t>(i) * ldb;
        double *dr = D + static_cast<size_t>(i) * ldd;
        for (int j = 0; j < c; j++) dr[j] = ar[j] + sign * br[j];
    }
}

// C(m x n) = A(m x k) * B(k x n) using Strassen for `levels` top levels.
static void strassen(int m, int k, int n,
                     const double *A, int lda,
                     const double *B, int ldb,
                     double *C, int ldc, int levels) {
    if (levels == 0) {
        base_mul(m, k, n, A, lda, B, ldb, C, ldc);
        return;
    }

    const int m2 = m / 2, k2 = k / 2, n2 = n / 2;

    const double *A11 = A;
    const double *A12 = A + k2;
    const double *A21 = A + static_cast<size_t>(m2) * lda;
    const double *A22 = A + static_cast<size_t>(m2) * lda + k2;

    const double *B11 = B;
    const double *B12 = B + n2;
    const double *B21 = B + static_cast<size_t>(k2) * ldb;
    const double *B22 = B + static_cast<size_t>(k2) * ldb + n2;

    double *C11 = C;
    double *C12 = C + n2;
    double *C21 = C + static_cast<size_t>(m2) * ldc;
    double *C22 = C + static_cast<size_t>(m2) * ldc + n2;

    std::vector<double> M1(static_cast<size_t>(m2) * n2);
    std::vector<double> M2(static_cast<size_t>(m2) * n2);
    std::vector<double> M3(static_cast<size_t>(m2) * n2);
    std::vector<double> M4(static_cast<size_t>(m2) * n2);
    std::vector<double> M5(static_cast<size_t>(m2) * n2);
    std::vector<double> M6(static_cast<size_t>(m2) * n2);
    std::vector<double> M7(static_cast<size_t>(m2) * n2);

    std::vector<double> TA(static_cast<size_t>(m2) * k2);
    std::vector<double> TB(static_cast<size_t>(k2) * n2);

    // M1 = (A11 + A22) (B11 + B22)
    madd(m2, k2, A11, lda, A22, lda, +1.0, TA.data(), k2);
    madd(k2, n2, B11, ldb, B22, ldb, +1.0, TB.data(), n2);
    strassen(m2, k2, n2, TA.data(), k2, TB.data(), n2, M1.data(), n2, levels - 1);

    // M2 = (A21 + A22) B11
    madd(m2, k2, A21, lda, A22, lda, +1.0, TA.data(), k2);
    strassen(m2, k2, n2, TA.data(), k2, B11, ldb, M2.data(), n2, levels - 1);

    // M3 = A11 (B12 - B22)
    madd(k2, n2, B12, ldb, B22, ldb, -1.0, TB.data(), n2);
    strassen(m2, k2, n2, A11, lda, TB.data(), n2, M3.data(), n2, levels - 1);

    // M4 = A22 (B21 - B11)
    madd(k2, n2, B21, ldb, B11, ldb, -1.0, TB.data(), n2);
    strassen(m2, k2, n2, A22, lda, TB.data(), n2, M4.data(), n2, levels - 1);

    // M5 = (A11 + A12) B22
    madd(m2, k2, A11, lda, A12, lda, +1.0, TA.data(), k2);
    strassen(m2, k2, n2, TA.data(), k2, B22, ldb, M5.data(), n2, levels - 1);

    // M6 = (A21 - A11) (B11 + B12)
    madd(m2, k2, A21, lda, A11, lda, -1.0, TA.data(), k2);
    madd(k2, n2, B11, ldb, B12, ldb, +1.0, TB.data(), n2);
    strassen(m2, k2, n2, TA.data(), k2, TB.data(), n2, M6.data(), n2, levels - 1);

    // M7 = (A12 - A22) (B21 + B22)
    madd(m2, k2, A12, lda, A22, lda, -1.0, TA.data(), k2);
    madd(k2, n2, B21, ldb, B22, ldb, +1.0, TB.data(), n2);
    strassen(m2, k2, n2, TA.data(), k2, TB.data(), n2, M7.data(), n2, levels - 1);

    // Combine
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m2; i++) {
        const size_t o = static_cast<size_t>(i) * n2;
        double *c11 = C11 + static_cast<size_t>(i) * ldc;
        double *c12 = C12 + static_cast<size_t>(i) * ldc;
        double *c21 = C21 + static_cast<size_t>(i) * ldc;
        double *c22 = C22 + static_cast<size_t>(i) * ldc;
        for (int j = 0; j < n2; j++) {
            const double m1 = M1[o + j], m2v = M2[o + j], m3 = M3[o + j];
            const double m4 = M4[o + j], m5 = M5[o + j], m6 = M6[o + j], m7 = M7[o + j];
            c11[j] = m1 + m4 - m5 + m7;
            c12[j] = m3 + m5;
            c21[j] = m2v + m4;
            c22[j] = m1 - m2v + m3 + m6;
        }
    }
}

void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int LEVELS = 2;
    constexpr int PADM = (1 << LEVELS) * 48;   // M divisible so base m%6==0, n%8==0
    constexpr int PADK = (1 << LEVELS);

    const int M = ((ny + PADM - 1) / PADM) * PADM;   // padded rows/cols (square)
    const int K = ((nx + PADK - 1) / PADK) * PADK;   // padded depth

    // A = normalized data, padded to M x K (extra rows/cols are zero).
    std::vector<double> A(static_cast<size_t>(M) * K, 0.0);
    // B = A^T, padded to K x M.
    std::vector<double> B(static_cast<size_t>(K) * M, 0.0);

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

        double *ar = &A[static_cast<size_t>(i) * K];
        for (int x = 0; x < nx; x++) {
            double val = (static_cast<double>(row[x]) - mean) * inv;
            ar[x] = val;
            B[static_cast<size_t>(x) * M + i] = val;
        }
    }

    std::vector<double> C(static_cast<size_t>(M) * M);
    strassen(M, K, M, A.data(), K, B.data(), M, C.data(), M, LEVELS);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; j++) {
        const double *cr = &C[static_cast<size_t>(j) * M];
        for (int i = j; i < ny; i++) {
            result[i + static_cast<size_t>(j) * ny] = static_cast<float>(cr[i]);
        }
    }
}
