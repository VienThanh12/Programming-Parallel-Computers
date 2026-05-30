#include "ppc.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_stats_kernel(const float* data, float* means, float* inv_stds, int ny, int nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ny) return;

    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (int j = 0; j < nx; ++j) {
        double v = (double)data[i * nx + j];
        sum += v;
        sum_sq += v * v;
    }

    double mean = sum / nx;
    double var = sum_sq / nx - mean * mean;
    double std = var > 0.0 ? sqrt(var) : 0.0;

    means[i] = (float)mean;
    inv_stds[i] = (std > 1e-9) ? (float)(1.0 / (std * sqrt((double)nx))) : 0.0f;
}

__global__ void pad_transpose_normalize_kernel(const float* in, float* t, const float* means, const float* inv_stds, int ny, int nx, int nn, int nx_pad) {
    int i = blockIdx.y;
    int ja = threadIdx.x;
    
    float mean_i = (i < ny) ? means[i] : 0.0f;
    float inv_std_i = (i < ny) ? inv_stds[i] : 0.0f;

    for (int jb = 0; jb < nx_pad; jb += 64) {
        int j = jb + ja;
        if (j < nx_pad) {
            float v = 0.0f;
            if (i < ny && j < nx) {
                v = (in[i * nx + j] - mean_i) * inv_std_i;
            }
            t[j * nn + i] = v; 
        }
    }
}

__global__ void correlate_kernel(float* result, const float* t, int ny, int nn, int nx_pad) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    __shared__ float xx[4][64];
    __shared__ float yy[4][64];

    float corr[8][8] = {0};

    int ija = ja * 8 + ia;
    int i = ic * 64 + ija;
    int j = jc * 64 + ija;

    for (int ks = 0; ks < nx_pad; ks += 4) {
        #pragma unroll
        for (int f = 0; f < 4; ++f) {
            int k = ks + f;
            xx[f][ija] = t[k * nn + i];
            yy[f][ija] = t[k * nn + j]; 
        }
        __syncthreads();

        #pragma unroll
        for (int f = 0; f < 4; ++f) {
            float x[8], y[8];
            #pragma unroll
            for (int ib = 0; ib < 8; ++ib) x[ib] = xx[f][ib * 8 + ia];
            #pragma unroll
            for (int jb = 0; jb < 8; ++jb) y[jb] = yy[f][jb * 8 + ja];

            #pragma unroll
            for (int ib = 0; ib < 8; ++ib) {
                #pragma unroll
                for (int jb = 0; jb < 8; ++jb) {
                    corr[ib][jb] += x[ib] * y[jb];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        int row_i = ic * 64 + ib * 8 + ia;
        if (row_i >= ny) continue;

        #pragma unroll
        for (int jb = 0; jb < 8; ++jb) {
            int row_j = jc * 64 + jb * 8 + ja;
            if (row_j >= ny || row_j > row_i) continue;
            
            float val = corr[ib][jb];
            
            if (row_i == row_j) {
                val = 1.0f;
            } else {
                if (val > 1.0f) val = 1.0f;
                if (val < -1.0f) val = -1.0f;
            }
            
            result[row_i + row_j * ny] = val;
        }
    }
}

void correlate(int ny, int nx, const float *data, float *result) {
    int nn = ((ny + 63) / 64) * 64;
    int nx_pad = ((nx + 3) / 4) * 4;

    float *d_data = nullptr, *d_t = nullptr, *d_result = nullptr;
    float *d_means = nullptr, *d_inv_stds = nullptr;
    
    PPC_CUDA_CHECK(cudaMalloc(&d_data, ny * nx * sizeof(float)), "cudaMalloc d_data");
    PPC_CUDA_CHECK(cudaMalloc(&d_t, nx_pad * nn * sizeof(float)), "cudaMalloc d_t");
    PPC_CUDA_CHECK(cudaMalloc(&d_result, ny * ny * sizeof(float)), "cudaMalloc d_result");
    PPC_CUDA_CHECK(cudaMalloc(&d_means, ny * sizeof(float)), "cudaMalloc d_means");
    PPC_CUDA_CHECK(cudaMalloc(&d_inv_stds, ny * sizeof(float)), "cudaMalloc d_inv_stds");
    
    PPC_CUDA_CHECK(cudaMemset(d_result, 0, ny * ny * sizeof(float)), "cudaMemset d_result");
    
    PPC_CUDA_CHECK(cudaMemcpy(d_data, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy data");

    dim3 stats_block(256);
    dim3 stats_grid((ny + 255) / 256);
    compute_stats_kernel<<<stats_grid, stats_block>>>(d_data, d_means, d_inv_stds, ny, nx);
    PPC_CUDA_CHECK(cudaGetLastError(), "stats launch");
    PPC_CUDA_CHECK(cudaDeviceSynchronize(), "stats sync");

    dim3 block1(64, 1), grid1(1, nn);
    pad_transpose_normalize_kernel<<<grid1, block1>>>(d_data, d_t, d_means, d_inv_stds, ny, nx, nn, nx_pad);
    PPC_CUDA_CHECK(cudaGetLastError(), "pad launch");
    PPC_CUDA_CHECK(cudaDeviceSynchronize(), "pad sync");

    dim3 block2(8, 8), grid2(nn / 64, nn / 64);
    correlate_kernel<<<grid2, block2>>>(d_result, d_t, ny, nn, nx_pad);
    PPC_CUDA_CHECK(cudaGetLastError(), "correlate launch");
    PPC_CUDA_CHECK(cudaDeviceSynchronize(), "correlate sync");

    PPC_CUDA_CHECK(cudaMemcpy(result, d_result, ny * ny * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy result");
    
    PPC_CUDA_CHECK(cudaFree(d_data), "cudaFree d_data");
    PPC_CUDA_CHECK(cudaFree(d_t), "cudaFree d_t");
    PPC_CUDA_CHECK(cudaFree(d_result), "cudaFree d_result");
    PPC_CUDA_CHECK(cudaFree(d_means), "cudaFree d_means");
    PPC_CUDA_CHECK(cudaFree(d_inv_stds), "cudaFree d_inv_stds");
}