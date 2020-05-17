
#include "matrix_defs.h"
#include "cuda_elim.h"

__global__ cuda_sub_mul_kernel(int size,
															 floating_type* matrix,
															 floating_type* vector)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

	MATRIX_PUT(a, size, i, j, 1);
	b[i] = 2;
}

void cuda_subtract_multiples(const int size,
							 						   floating_type* matrix,
							 							 floating_type* vector)
{
	//double card_mat[size][size];
	//double card_vect[size];
	//double card_output[size];

	dim3 threads_per_block(100,100);
	dim3 num_blocks(size / threads_per_block.x, size / threads_per_block.y);
	cuda_sub_mul_kernel<<<num_blocks, threads_per_block>>>(size, matrix, vector);
}