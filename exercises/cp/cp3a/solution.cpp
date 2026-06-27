#include <cmath>
#include <vector>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

constexpr double4_t d4zero {0.0, 0.0, 0.0, 0.0};

void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int nb = 4;                       // rows packed per vector
    const int nyb = (ny + nb - 1) / nb;         // number of 4-row blocks

    // T[jb * nx + x] holds {X[4jb+0][x], X[4jb+1][x], X[4jb+2][x], X[4jb+3][x]}
    // (normalized data, transposed so that 4 consecutive rows share a vector).
    std::vector<double4_t> T(static_cast<size_t>(nyb) * nx, d4zero);

    // ---- Normalize rows and pack into T ----
    #pragma omp parallel for schedule(static)
    for (int jb = 0; jb < nyb; jb++) {
        for (int lane = 0; lane < nb; lane++) {
            const int i = jb * nb + lane;
            if (i >= ny) continue;              // padded rows stay zero

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

            double4_t *dst = &T[static_cast<size_t>(jb) * nx];
            for (int x = 0; x < nx; x++) {
                dst[x][lane] = (static_cast<double>(row[x]) - mean) * inv;
            }
        }
    }

    // Store a 4x4 block of correlations: result[i + j*ny] for j <= i.
    auto store = [&](int ib, int jb, double4_t r0, double4_t r1,
                     double4_t r2, double4_t r3) {
        const double4_t rr[4] = {r0, r1, r2, r3};
        for (int a = 0; a < nb; a++) {
            const int i = ib * nb + a;
            if (i >= ny) continue;
            for (int b = 0; b < nb; b++) {
                const int j = jb * nb + b;
                if (j < ny && j <= i) {
                    result[i + static_cast<size_t>(j) * ny] =
                        static_cast<float>(rr[a][b]);
                }
            }
        }
    };

    // ---- Compute X * X^T using packed rows (outer-product micro-kernel) ----
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ib = 0; ib < nyb; ib++) {
        const double4_t *Ti = &T[static_cast<size_t>(ib) * nx];

        int jb = 0;
        // Process two j-blocks at a time: 4 i-rows x 8 j-rows per inner step.
        for (; jb + 1 <= ib; jb += 2) {
            const double4_t *Tj0 = &T[static_cast<size_t>(jb) * nx];
            const double4_t *Tj1 = &T[static_cast<size_t>(jb + 1) * nx];

            double4_t a0 = d4zero, a1 = d4zero, a2 = d4zero, a3 = d4zero;
            double4_t c0 = d4zero, c1 = d4zero, c2 = d4zero, c3 = d4zero;

            for (int k = 0; k < nx; k++) {
                double4_t vi = Ti[k];
                double4_t u0 = Tj0[k];
                double4_t u1 = Tj1[k];
                double4_t b0 = {vi[0], vi[0], vi[0], vi[0]};
                double4_t b1 = {vi[1], vi[1], vi[1], vi[1]};
                double4_t b2 = {vi[2], vi[2], vi[2], vi[2]};
                double4_t b3 = {vi[3], vi[3], vi[3], vi[3]};
                a0 += b0 * u0; a1 += b1 * u0; a2 += b2 * u0; a3 += b3 * u0;
                c0 += b0 * u1; c1 += b1 * u1; c2 += b2 * u1; c3 += b3 * u1;
            }

            store(ib, jb, a0, a1, a2, a3);
            store(ib, jb + 1, c0, c1, c2, c3);
        }

        // Leftover diagonal block (jb == ib) when the block count is odd.
        if (jb <= ib) {
            const double4_t *Tj0 = &T[static_cast<size_t>(jb) * nx];
            double4_t a0 = d4zero, a1 = d4zero, a2 = d4zero, a3 = d4zero;

            for (int k = 0; k < nx; k++) {
                double4_t vi = Ti[k];
                double4_t u0 = Tj0[k];
                a0 += (double4_t){vi[0], vi[0], vi[0], vi[0]} * u0;
                a1 += (double4_t){vi[1], vi[1], vi[1], vi[1]} * u0;
                a2 += (double4_t){vi[2], vi[2], vi[2], vi[2]} * u0;
                a3 += (double4_t){vi[3], vi[3], vi[3], vi[3]} * u0;
            }

            store(ib, jb, a0, a1, a2, a3);
        }
    }
}
