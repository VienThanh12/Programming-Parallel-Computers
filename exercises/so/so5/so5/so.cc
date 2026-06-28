#include <vector>
#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

void psort(int n, data_t *data) {
    if (n <= 1) return;

    int threads = omp_get_max_threads();

    int blocks = 1;
    while (blocks < threads) blocks <<= 1;
    if (blocks > n) blocks = 1;

    std::vector<int> bound(blocks + 1);
    for (int b = 0; b <= blocks; ++b) {
        bound[b] = static_cast<int>(static_cast<long long>(n) * b / blocks);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < blocks; ++b) {
        std::sort(data + bound[b], data + bound[b + 1]);
    }

    for (int width = 1; width < blocks; width <<= 1) {
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < blocks; b += 2 * width) {
            int lo = bound[b];
            int mid = bound[std::min(b + width, blocks)];
            int hi = bound[std::min(b + 2 * width, blocks)];
            if (mid < hi) {
                std::inplace_merge(data + lo, data + mid, data + hi);
            }
        }
    }
}