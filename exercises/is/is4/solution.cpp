#include <vector>
#include <cstddef>
#include <immintrin.h>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

Result segment(int ny, int nx, const float *data) {
    const int nx1 = nx + 1;
    const int ny1 = ny + 1;
    const size_t n = static_cast<size_t>(ny1) * nx1;

    // Double-precision prefix sums, one array per color channel.
    std::vector<double> S0(n, 0.0), S1(n, 0.0), S2(n, 0.0);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const size_t i = static_cast<size_t>(y + 1) * nx1 + (x + 1);
            const size_t up = static_cast<size_t>(y) * nx1 + (x + 1);
            const size_t lf = static_cast<size_t>(y + 1) * nx1 + x;
            const size_t ul = static_cast<size_t>(y) * nx1 + x;
            const size_t p = static_cast<size_t>(3) * (x + nx * y);
            S0[i] = data[p + 0] + S0[up] + S0[lf] - S0[ul];
            S1[i] = data[p + 1] + S1[up] + S1[lf] - S1[ul];
            S2[i] = data[p + 2] + S2[up] + S2[lf] - S2[ul];
        }
    }

    const double P = static_cast<double>(ny) * nx;
    const double A0 = S0[static_cast<size_t>(ny) * nx1 + nx];
    const double A1 = S1[static_cast<size_t>(ny) * nx1 + nx];
    const double A2 = S2[static_cast<size_t>(ny) * nx1 + nx];
    const double Asq = A0 * A0 + A1 * A1 + A2 * A2;

    double globalBest = -1.0;
    Result best{0, 0, 1, 1, {0, 0, 0}, {0, 0, 0}};

    #pragma omp parallel
    {
        double localBest = -1.0;
        int bY0 = 0, bX0 = 0, bY1 = 1, bX1 = 1;
        std::vector<double> dr(nx1), dg(nx1), db(nx1), pc(nx1);
        std::vector<double> coefA(nx1), coefB(nx1);

        #pragma omp for schedule(dynamic, 1)
        for (int h = 1; h <= ny; ++h) {
            // Size-only coefficients: depend on (h, w) but not on position.
            for (int w = 1; w <= nx; ++w) {
                const double nIn = static_cast<double>(h) * w;
                const double nOut = P - nIn;
                if (nOut > 0.0) {
                    coefA[w] = 1.0 / nIn + 1.0 / nOut;
                    coefB[w] = 1.0 / nOut;
                } else {
                    coefA[w] = 1.0 / nIn;
                    coefB[w] = 0.0;
                }
            }

            for (int y0 = 0; y0 + h <= ny; ++y0) {
                const int y1 = y0 + h;
                const double *r1a = &S0[static_cast<size_t>(y1) * nx1];
                const double *r0a = &S0[static_cast<size_t>(y0) * nx1];
                const double *r1b = &S1[static_cast<size_t>(y1) * nx1];
                const double *r0b = &S1[static_cast<size_t>(y0) * nx1];
                const double *r1c = &S2[static_cast<size_t>(y1) * nx1];
                const double *r0c = &S2[static_cast<size_t>(y0) * nx1];
                for (int x = 0; x <= nx; ++x) {
                    const double vr = r1a[x] - r0a[x];
                    const double vg = r1b[x] - r0b[x];
                    const double vb = r1c[x] - r0c[x];
                    dr[x] = vr;
                    dg[x] = vg;
                    db[x] = vb;
                    pc[x] = A0 * vr + A1 * vg + A2 * vb;
                }

                const __m256d vAsq = _mm256_set1_pd(Asq);
                const __m256d vtwo = _mm256_set1_pd(2.0);
                const __m256d vstep = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);

                for (int x0 = 0; x0 < nx; ++x0) {
                    const __m256d vr0 = _mm256_set1_pd(dr[x0]);
                    const __m256d vg0 = _mm256_set1_pd(dg[x0]);
                    const __m256d vb0 = _mm256_set1_pd(db[x0]);
                    const __m256d vpc0 = _mm256_set1_pd(pc[x0]);
                    __m256d vbestf = _mm256_set1_pd(-1.0);
                    __m256d vbestx = _mm256_setzero_pd();

                    int x1 = x0 + 1;
                    for (; x1 + 4 <= nx + 1; x1 += 4) {
                        const __m256d sr =
                            _mm256_sub_pd(_mm256_loadu_pd(&dr[x1]), vr0);
                        const __m256d sg =
                            _mm256_sub_pd(_mm256_loadu_pd(&dg[x1]), vg0);
                        const __m256d sb =
                            _mm256_sub_pd(_mm256_loadu_pd(&db[x1]), vb0);
                        const __m256d Q = _mm256_add_pd(
                            _mm256_add_pd(_mm256_mul_pd(sr, sr),
                                          _mm256_mul_pd(sg, sg)),
                            _mm256_mul_pd(sb, sb));
                        const __m256d dot =
                            _mm256_sub_pd(_mm256_loadu_pd(&pc[x1]), vpc0);
                        const __m256d cA = _mm256_loadu_pd(&coefA[x1 - x0]);
                        const __m256d cB = _mm256_loadu_pd(&coefB[x1 - x0]);
                        // f = cA*Q + cB*(Asq - 2*dot)
                        const __m256d f = _mm256_add_pd(
                            _mm256_mul_pd(cA, Q),
                            _mm256_mul_pd(
                                cB, _mm256_sub_pd(
                                        vAsq, _mm256_mul_pd(vtwo, dot))));
                        const __m256d xidx = _mm256_add_pd(
                            _mm256_set1_pd(static_cast<double>(x1)), vstep);
                        const __m256d mask =
                            _mm256_cmp_pd(f, vbestf, _CMP_GT_OQ);
                        vbestf = _mm256_blendv_pd(vbestf, f, mask);
                        vbestx = _mm256_blendv_pd(vbestx, xidx, mask);
                    }

                    double fa[4], xa[4];
                    _mm256_storeu_pd(fa, vbestf);
                    _mm256_storeu_pd(xa, vbestx);
                    double bf = -1.0;
                    int bx = x0 + 1;
                    for (int k = 0; k < 4; ++k) {
                        if (fa[k] > bf) {
                            bf = fa[k];
                            bx = static_cast<int>(xa[k]);
                        }
                    }

                    const double r0 = dr[x0], g0 = dg[x0], b0 = db[x0];
                    const double pc0 = pc[x0];
                    for (; x1 <= nx; ++x1) {
                        const int w = x1 - x0;
                        const double sr = dr[x1] - r0;
                        const double sg = dg[x1] - g0;
                        const double sb = db[x1] - b0;
                        const double Q = sr * sr + sg * sg + sb * sb;
                        const double dot = pc[x1] - pc0;
                        const double f =
                            coefA[w] * Q + coefB[w] * (Asq - 2.0 * dot);
                        if (f > bf) {
                            bf = f;
                            bx = x1;
                        }
                    }

                    if (bf > localBest) {
                        localBest = bf;
                        bY0 = y0;
                        bX0 = x0;
                        bY1 = y1;
                        bX1 = bx;
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (localBest > globalBest) {
                globalBest = localBest;
                best.y0 = bY0;
                best.x0 = bX0;
                best.y1 = bY1;
                best.x1 = bX1;
            }
        }
    }

    // Recover the inner/outer colors for the winning rectangle.
    auto box = [&](const std::vector<double> &S, int y0, int x0, int y1,
                   int x1) -> double {
        return S[static_cast<size_t>(y1) * nx1 + x1]
             - S[static_cast<size_t>(y0) * nx1 + x1]
             - S[static_cast<size_t>(y1) * nx1 + x0]
             + S[static_cast<size_t>(y0) * nx1 + x0];
    };

    const double nIn =
        static_cast<double>(best.y1 - best.y0) * (best.x1 - best.x0);
    const double nOut = P - nIn;
    const double in0 = box(S0, best.y0, best.x0, best.y1, best.x1);
    const double in1 = box(S1, best.y0, best.x0, best.y1, best.x1);
    const double in2 = box(S2, best.y0, best.x0, best.y1, best.x1);
    best.inner[0] = static_cast<float>(in0 / nIn);
    best.inner[1] = static_cast<float>(in1 / nIn);
    best.inner[2] = static_cast<float>(in2 / nIn);
    best.outer[0] = static_cast<float>(nOut > 0.0 ? (A0 - in0) / nOut : 0.0);
    best.outer[1] = static_cast<float>(nOut > 0.0 ? (A1 - in1) / nOut : 0.0);
    best.outer[2] = static_cast<float>(nOut > 0.0 ? (A2 - in2) / nOut : 0.0);

    return best;
}
