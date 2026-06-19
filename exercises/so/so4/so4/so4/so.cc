#include <algorithm>
#include <memory>
#include <cstddef>
#include <omp.h>

typedef unsigned long long data_t;

static const std::size_t SORT_CUTOFF = 1 << 15; 
static const std::size_t MERGE_CUTOFF = 1 << 14;

static void parallel_merge(const data_t *a, std::size_t na,
                           const data_t *b, std::size_t nb,
                           data_t *out, int depth) {
    if (depth <= 0 || na + nb <= MERGE_CUTOFF) {
        std::merge(a, a + na, b, b + nb, out);
        return;
    }
    if (na < nb) {
        std::swap(a, b);
        std::swap(na, nb);
    }
    std::size_t ma = na / 2;
    std::size_t mb = std::lower_bound(b, b + nb, a[ma]) - b;

    #pragma omp task shared(a, b, out)
    parallel_merge(a, ma, b, mb, out, depth - 1);
    parallel_merge(a + ma, na - ma, b + mb, nb - mb, out + ma + mb, depth - 1);
    #pragma omp taskwait
}

static void merge_sort(data_t *src, data_t *buf, std::size_t n, int depth,
                       bool result_in_src) {
    if (depth <= 0 || n <= SORT_CUTOFF) {
        std::sort(src, src + n);
        if (!result_in_src) {
            std::copy(src, src + n, buf);
        }
        return;
    }

    std::size_t m = n / 2;
    bool child_in_src = !result_in_src;

    #pragma omp task shared(src, buf)
    merge_sort(src, buf, m, depth - 1, child_in_src);
    merge_sort(src + m, buf + m, n - m, depth - 1, child_in_src);
    #pragma omp taskwait

    if (child_in_src) {
        parallel_merge(src, m, src + m, n - m, buf, depth);
    } else {
        parallel_merge(buf, m, buf + m, n - m, src, depth);
    }
}

void psort(int n, data_t *data) {
    if (n <= 1) {
        return;
    }


    std::unique_ptr<data_t[]> buf(new data_t[(std::size_t)n]);

    int depth = 0;
    int threads = omp_get_max_threads();
    while ((1 << depth) < threads * 4) {
        depth++;
    }

    #pragma omp parallel
    {
        #pragma omp single
        merge_sort(data, buf.get(), (std::size_t)n, depth, true);
    }
}