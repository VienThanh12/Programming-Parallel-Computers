#include <iostream>
#include <vector>
#include <omp.h> 
typedef unsigned long long data_t;

void merge_sort_helper(data_t* data, data_t* temp, int left, int right) {
    if (right - left <= 1) {
        return;
    }

    int mid = left + (right - left) / 2;

    // Create tasks for the recursive calls. 
    // The 'if' clause prevents creating too many tasks for tiny subarrays, 
    // which would slow things down due to overhead.
    #pragma omp task shared(data, temp) if(right - left > 10000)
    merge_sort_helper(data, temp, left, mid);

    #pragma omp task shared(data, temp) if(right - left > 10000)
    merge_sort_helper(data, temp, mid, right);

    // Wait for both halves to finish sorting before merging
    #pragma omp taskwait

    // The merge step MUST remain sequential
    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        if (data[i] <= data[j]) {
            temp[k++] = data[i++];
        } else {
            temp[k++] = data[j++];
        }
    }

    while (i < mid) {
        temp[k++] = data[i++];
    }

    while (j < right) {
        temp[k++] = data[j++];
    }

    for (int p = left; p < right; ++p) {
        data[p] = temp[p];
    }
}

void psort(int n, data_t *data) {
    if (n <= 1) return;

    std::vector<data_t> temp(n);

    // Start a single parallel region at the very beginning
    #pragma omp parallel
    {
        // Ensure only one thread kicks off the initial recursive call
        #pragma omp single
        {
            merge_sort_helper(data, temp.data(), 0, n);
        }
    }
}