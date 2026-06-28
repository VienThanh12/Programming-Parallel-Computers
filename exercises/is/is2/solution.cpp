#include <vector>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    // Prefix sums S[y][x][c] = sum of data over all pixels (yy, xx) with
    // yy < y and xx < x. Dimensions (ny+1) x (nx+1) x 3, with a zero border.
    std::vector<double> S(static_cast<size_t>(ny + 1) * (nx + 1) * 3, 0.0);
    auto idx = [&](int y, int x, int c) -> size_t {
        return (static_cast<size_t>(y) * (nx + 1) + x) * 3 + c;
    };

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            for (int c = 0; c < 3; ++c) {
                double v = data[c + 3 * x + 3 * nx * y];
                S[idx(y + 1, x + 1, c)] =
                    v + S[idx(y, x + 1, c)] + S[idx(y + 1, x, c)] - S[idx(y, x, c)];
            }
        }
    }

    const double totalPixels = static_cast<double>(ny) * nx;
    double sumAll[3];
    for (int c = 0; c < 3; ++c) {
        sumAll[c] = S[idx(ny, nx, c)];
    }

    double bestScore = -1.0;
    Result best{0, 0, 1, 1, {0, 0, 0}, {0, 0, 0}};

    // Try every rectangle [y0, y1) x [x0, x1) and pick the one with the
    // maximum "score" = sum_c (sum_in[c]^2 / nIn + sum_out[c]^2 / nOut),
    // which is equivalent to minimizing the total sum of squared errors.
    for (int y0 = 0; y0 < ny; ++y0) {
        for (int y1 = y0 + 1; y1 <= ny; ++y1) {
            for (int x0 = 0; x0 < nx; ++x0) {
                for (int x1 = x0 + 1; x1 <= nx; ++x1) {
                    double nIn = static_cast<double>(y1 - y0) * (x1 - x0);
                    double nOut = totalPixels - nIn;

                    double score = 0.0;
                    for (int c = 0; c < 3; ++c) {
                        double sumIn = S[idx(y1, x1, c)] - S[idx(y0, x1, c)] -
                                       S[idx(y1, x0, c)] + S[idx(y0, x0, c)];
                        double sumOut = sumAll[c] - sumIn;
                        score += sumIn * sumIn / nIn;
                        if (nOut > 0.0) {
                            score += sumOut * sumOut / nOut;
                        }
                    }

                    if (score > bestScore) {
                        bestScore = score;
                        best.y0 = y0;
                        best.x0 = x0;
                        best.y1 = y1;
                        best.x1 = x1;
                        for (int c = 0; c < 3; ++c) {
                            double sumIn = S[idx(y1, x1, c)] - S[idx(y0, x1, c)] -
                                           S[idx(y1, x0, c)] + S[idx(y0, x0, c)];
                            double sumOut = sumAll[c] - sumIn;
                            best.inner[c] = static_cast<float>(sumIn / nIn);
                            best.outer[c] =
                                static_cast<float>(nOut > 0.0 ? sumOut / nOut : 0.0);
                        }
                    }
                }
            }
        }
    }

    return best;
}
