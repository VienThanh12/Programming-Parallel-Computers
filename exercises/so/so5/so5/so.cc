#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h> 

typedef unsigned long long data_t;

int partition(data_t* data, int left, int right) {
    int mid = left + (right - left) / 2;
    if (data[mid] < data[left]) std::swap(data[left], data[mid]);
    if (data[right] < data[left]) std::swap(data[left], data[right]);
    if (data[right] < data[mid]) std::swap(data[mid], data[right]);
    
    std::swap(data[mid], data[right]); 
    data_t pivot = data[right];
    
    int i = left - 1;
    for (int j = left; j < right; ++j) {
        if (data[j] < pivot) {
            ++i;
            std::swap(data[i], data[j]);
        }
    }
    std::swap(data[i + 1], data[right]);
    return i + 1;
}

void quicksort_helper(data_t* data, int left, int right) {
    if (left >= right) {
        return;
    }

    if (right - left < 10000) {
        std::sort(data + left, data + right + 1);
        return;
    }

    int pivot_idx = partition(data, left, right);

    #pragma omp task shared(data)
    quicksort_helper(data, left, pivot_idx - 1);

    #pragma omp task shared(data)
    quicksort_helper(data, pivot_idx + 1, right);

    #pragma omp taskwait
}

void psort(int n, data_t *data) {
    if (n <= 1) return;

    #pragma omp parallel
    {
        #pragma omp single
        {
            quicksort_helper(data, 0, n - 1);
        }
    }
}