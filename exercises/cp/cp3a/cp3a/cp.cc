#include <cmath>
#include <vector>
#include <algorithm>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

constexpr double4_t d4zero {0.0, 0.0, 0.0, 0.0};

static inline double hsum4(double4_t v) {
    return v[0] + v[1] + v[2] + v[3];
}

void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int nb = 4;
    const int na = (nx + nb - 1) / nb;

    std::vector<double4_t> nor(static_cast<size_t>(ny) * na, d4zero);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; i++) {
        const float *row = &data[i * nx];
        double mean = 0.0;
        for (int x = 0; x < nx; x++) {
            mean += row[x];
        }
        mean /= static_cast<double>(nx);

        double ss = 0.0;
        for (int x = 0; x < nx; x++) {
            double v = static_cast<double>(row[x]) - mean;
            ss += v * v;
        }
        double inv = (ss > 0.0) ? 1.0 / std::sqrt(ss) : 0.0;

        double *dst = reinterpret_cast<double *>(&nor[static_cast<size_t>(i) * na]);
        for (int x = 0; x < nx; x++) {
            dst[x] = (static_cast<double>(row[x]) - mean) * inv;
        }
        for (int x = nx; x < na * nb; x++) {
            dst[x] = 0.0;
        }
    }

    constexpr int nd = 3;
    const int nc = (ny + nd - 1) / nd;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int ic = 0; ic < nc; ic++) {
        for (int jc = 0; jc <= ic; jc++) {
            const int ir = ic * nd;
            const int jr = jc * nd;

            const double4_t *ri[nd];
            const double4_t *rj[nd];
            for (int id = 0; id < nd; id++) {
                int r = std::min(ir + id, ny - 1);
                ri[id] = &nor[static_cast<size_t>(r) * na];
            }
            for (int jd = 0; jd < nd; jd++) {
                int r = std::min(jr + jd, ny - 1);
                rj[jd] = &nor[static_cast<size_t>(r) * na];
            }

            double4_t vsum[nd][nd];
            for (int id = 0; id < nd; id++) {
                for (int jd = 0; jd < nd; jd++) {
                    vsum[id][jd] = d4zero;
                }
            }

            for (int k = 0; k < na; k++) {
                double4_t x0 = ri[0][k];
                double4_t x1 = ri[1][k];
                double4_t x2 = ri[2][k];
                double4_t y0 = rj[0][k];
                double4_t y1 = rj[1][k];
                double4_t y2 = rj[2][k];
                vsum[0][0] += x0 * y0; vsum[0][1] += x0 * y1; vsum[0][2] += x0 * y2;
                vsum[1][0] += x1 * y0; vsum[1][1] += x1 * y1; vsum[1][2] += x1 * y2;
                vsum[2][0] += x2 * y0; vsum[2][1] += x2 * y1; vsum[2][2] += x2 * y2;
            }

            for (int id = 0; id < nd; id++) {
                int i = ir + id;
                if (i >= ny) continue;
                for (int jd = 0; jd < nd; jd++) {
                    int j = jr + jd;
                    if (j < ny && j <= i) {
                        result[i + static_cast<size_t>(j) * ny] = static_cast<float>(hsum4(vsum[id][jd]));
                    }
                }
            }
        }
    }
}
