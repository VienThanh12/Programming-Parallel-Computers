#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

static const int SORT_CUTOFF = 50000;

void merge_sort(data_t* data, data_t* buf, int left, int right, int depth) {
    int len = right - left;
    if (len <= SORT_CUTOFF) {
        std::sort(data + left, data + right);
        return;
    }

    int mid = left + len / 2;

    if (depth > 0) {
        #pragma omp task shared(data, buf)
        merge_sort(data, buf, left, mid, depth - 1);

        merge_sort(data, buf, mid, right, depth - 1);

        #pragma omp taskwait
    } else {
        merge_sort(data, buf, left, mid, 0);
        merge_sort(data, buf, mid, right, 0);
    }

    std::merge(data + left, data + mid, data + mid, data + right, buf + left);
    std::copy(buf + left, buf + right, data + left);
}

void psort(int n, data_t *data) {
    if (n <= 1) return;

    std::vector<data_t> buf(n);

    int depth = 0;
    int threads = omp_get_max_threads();
    while ((1 << depth) < threads * 4) {
        depth++;
    }

    #pragma omp parallel
    {
        #pragma omp single
        merge_sort(data, buf.data(), 0, n, depth);
    }
}