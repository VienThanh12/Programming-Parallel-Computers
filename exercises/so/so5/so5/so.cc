#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h> 

typedef unsigned long long data_t;

void quicksort_helper(data_t* data, int left, int right, int depth) {
    if (left >= right) {
        return;
    }

    if (right - left < 100000) {
        std::sort(data + left, data + right + 1);
        return;
    }

    int mid = left + (right - left) / 2;
    if (data[mid] < data[left]) std::swap(data[left], data[mid]);
    if (data[right] < data[left]) std::swap(data[left], data[right]);
    if (data[right] < data[mid]) std::swap(data[mid], data[right]);
    
    data_t pivot = data[mid];

    int i = left;
    int j = left;
    int k = right;

    while (j <= k) {
        if (data[j] < pivot) {
            std::swap(data[i], data[j]);
            i++;
            j++;
        } else if (data[j] > pivot) {
            std::swap(data[j], data[k]);
            k--;
        } else {
            j++;
        }
    }

    if (depth <= 0) {
        quicksort_helper(data, left, i - 1, 0);
        quicksort_helper(data, k + 1, right, 0);
        return;
    }

    #pragma omp task shared(data)
    quicksort_helper(data, left, i - 1, depth - 1);

    #pragma omp task shared(data)
    quicksort_helper(data, k + 1, right, depth - 1);

    #pragma omp taskwait
}

void psort(int n, data_t *data) {
    if (n <= 1) return;

    int depth = 0;
    int threads = omp_get_max_threads();
    while ((1 << depth) < threads * 4) {
        depth++;
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            quicksort_helper(data, 0, n - 1, depth);
        }
    }
}