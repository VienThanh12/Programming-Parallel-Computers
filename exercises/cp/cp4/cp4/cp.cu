/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/
// Required for CUDA error checking macro
#include "ppc.h"
#include <cuda_runtime.h>

__global__ void correlate_kernel(int ny, int nx, const float *data, float *result) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ny || j > i) return;

	float mean_i = 0.0f, mean_j = 0.0f;
	float std_i = 0.0f, std_j = 0.0f;
	for (int x = 0; x < nx; ++x) {
		float v_i = data[x + i * nx];
		float v_j = data[x + j * nx];
		mean_i += v_i;
		mean_j += v_j;
	}
	mean_i /= nx;
	mean_j /= nx;
	for (int x = 0; x < nx; ++x) {
		float v_i = data[x + i * nx];
		float v_j = data[x + j * nx];
		std_i += (v_i - mean_i) * (v_i - mean_i);
		std_j += (v_j - mean_j) * (v_j - mean_j);
	}
	std_i = sqrtf(std_i);
	std_j = sqrtf(std_j);
	if (std_i == 0.0f || std_j == 0.0f) {
		result[i + j * ny] = 0.0f;
		return;
	}
	float corr = 0.0f;
	for (int x = 0; x < nx; ++x) {
		float v_i = data[x + i * nx];
		float v_j = data[x + j * nx];
		corr += (v_i - mean_i) * (v_j - mean_j);
	}
	corr /= (std_i * std_j);
	result[i + j * ny] = corr;
}

void correlate(int ny, int nx, const float *data, float *result) {
	float *d_data = nullptr, *d_result = nullptr;
	size_t data_size = sizeof(float) * ny * nx;
	size_t result_size = sizeof(float) * ny * ny;
	PPC_CUDA_CHECK(cudaMalloc(&d_data, data_size), "cudaMalloc d_data");
	PPC_CUDA_CHECK(cudaMalloc(&d_result, result_size), "cudaMalloc d_result");
	PPC_CUDA_CHECK(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice), "cudaMemcpy H2D data");

	// Zero out the device result array to avoid uninitialized memory errors
	PPC_CUDA_CHECK(cudaMemset(d_result, 0, result_size), "cudaMemset d_result");

	dim3 block(16, 16);
	dim3 grid((ny + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	correlate_kernel<<<grid, block>>>(ny, nx, d_data, d_result);
	PPC_CUDA_CHECK(cudaGetLastError(), "Kernel launch");
	PPC_CUDA_CHECK(cudaDeviceSynchronize(), "Kernel sync");

	PPC_CUDA_CHECK(cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H result");
	PPC_CUDA_CHECK(cudaFree(d_data), "cudaFree d_data");
	PPC_CUDA_CHECK(cudaFree(d_result), "cudaFree d_result");
}
