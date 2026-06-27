#include <cmath>
#include <vector>

typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));

constexpr float8_t f8zero {0, 0, 0, 0, 0, 0, 0, 0};

static inline float8_t bcast(const float *p) {
    return (float8_t){*p, *p, *p, *p, *p, *p, *p, *p};
}

void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int JB = 8;
    constexpr int MI = 6;
    constexpr int NJ = 2;

    const int nyb = (ny + JB - 1) / JB;
    const int nyi = ((ny + MI - 1) / MI) * MI;

    std::vector<float> Xr(static_cast<size_t>(nyi) * nx, 0.0f);
    std::vector<float8_t> Xt(static_cast<size_t>(nyb + NJ) * nx, f8zero);

    #pragma omp parallel for schedule(static)
    for (int jb = 0; jb < nyb; jb++) {
        for (int lane = 0; lane < JB; lane++) {
            const int i = jb * JB + lane;
            if (i >= ny) continue;

            const float *row = &data[static_cast<size_t>(i) * nx];
            double mean = 0.0;
            for (int x = 0; x < nx; x++) mean += row[x];
            const float m = static_cast<float>(mean / nx);

            double ss = 0.0;
            for (int x = 0; x < nx; x++) {
                double v = static_cast<double>(row[x]) - m;
                ss += v * v;
            }
            const float inv = (ss > 0.0) ? static_cast<float>(1.0 / std::sqrt(ss)) : 0.0f;

            float *xr = &Xr[static_cast<size_t>(i) * nx];
            float8_t *xt = &Xt[static_cast<size_t>(jb) * nx];
            for (int x = 0; x < nx; x++) {
                float val = (row[x] - m) * inv;
                xr[x] = val;
                xt[x][lane] = val;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i0 = 0; i0 < ny; i0 += MI) {
        const float *a0 = &Xr[static_cast<size_t>(i0 + 0) * nx];
        const float *a1 = &Xr[static_cast<size_t>(i0 + 1) * nx];
        const float *a2 = &Xr[static_cast<size_t>(i0 + 2) * nx];
        const float *a3 = &Xr[static_cast<size_t>(i0 + 3) * nx];
        const float *a4 = &Xr[static_cast<size_t>(i0 + 4) * nx];
        const float *a5 = &Xr[static_cast<size_t>(i0 + 5) * nx];

        const int jmax = i0 + MI - 1;
        for (int j0 = 0; j0 <= jmax; j0 += JB * NJ) {
            const float8_t *b0 = &Xt[static_cast<size_t>(j0 / JB) * nx];
            const float8_t *b1 = &Xt[static_cast<size_t>(j0 / JB + 1) * nx];

            float8_t r00 = f8zero, r01 = f8zero;
            float8_t r10 = f8zero, r11 = f8zero;
            float8_t r20 = f8zero, r21 = f8zero;
            float8_t r30 = f8zero, r31 = f8zero;
            float8_t r40 = f8zero, r41 = f8zero;
            float8_t r50 = f8zero, r51 = f8zero;

            for (int k = 0; k < nx; k++) {
                float8_t bb0 = b0[k];
                float8_t bb1 = b1[k];
                float8_t av;
                av = bcast(&a0[k]); r00 += av * bb0; r01 += av * bb1;
                av = bcast(&a1[k]); r10 += av * bb0; r11 += av * bb1;
                av = bcast(&a2[k]); r20 += av * bb0; r21 += av * bb1;
                av = bcast(&a3[k]); r30 += av * bb0; r31 += av * bb1;
                av = bcast(&a4[k]); r40 += av * bb0; r41 += av * bb1;
                av = bcast(&a5[k]); r50 += av * bb0; r51 += av * bb1;
            }

            const float8_t *acc[MI][NJ] = {
                {&r00, &r01}, {&r10, &r11}, {&r20, &r21},
                {&r30, &r31}, {&r40, &r41}, {&r50, &r51},
            };
            for (int m = 0; m < MI; m++) {
                const int i = i0 + m;
                if (i >= ny) continue;
                for (int n = 0; n < NJ; n++) {
                    const float8_t v = *acc[m][n];
                    for (int lane = 0; lane < JB; lane++) {
                        const int j = j0 + n * JB + lane;
                        if (j < ny && j <= i) {
                            result[i + static_cast<size_t>(j) * ny] = v[lane];
                        }
                    }
                }
            }
        }
    }
}
