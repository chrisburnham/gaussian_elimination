#include "cuda_elim.h"
#include <cuda_runtime.h>

__global__ void cuda_sub_mul_kernel(int size,
								     							  floating_type* matrix,
															      floating_type* vector,
															      int i)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if((j < size) || (k < size))
  {
	  floating_type m = MATRIX_GET( matrix, size, j, i ) / MATRIX_GET( matrix, size, i, i );
	  MATRIX_PUT(matrix,
	             size,
	             j,
	             k,
	             MATRIX_GET( matrix, size, j, k ) -
	                   m * MATRIX_GET( matrix, size, i, k ) );

	  if( k == size -1)
	  {
	  	vector[j] -= m * b[i];
	  }
	}
}

void cuda_subtract_multiples(const int size,
							 						   floating_type* matrix,
							 							 floating_type* vector,
							 							 int i)
{
	cudaError_t result;

	floating_type* dev_mat;
	floating_type* dev_vect;

	cudoMemcpy(dev_mat, matrix, size*size*sizeof(floating_type), cudaMemcpyHostToDevice);
	cudoMemcpy(dev_vect, vector, size*sizeof(floating_type), cudaMemcpyHostToDevice);

	dim3 threads_per_block(100,100);
	dim3 num_blocks(size / threads_per_block.x, size / threads_per_block.y);
	cuda_sub_mul_kernel<<<num_blocks, threads_per_block>>>(size, dev_mat, dev_vect, i);
	
	result = cudaGetLastError();
	if(result != cudaSuccess)
	{
		printf("Failure running kernel: %s\n", cudaGetErrorString(result));
	}

	cudoMemcpy(matrix, dev_mat, size*size*sizeof(floating_type), cudaMemcpyDeviceToHost);
	cudoMemcpy(vector, dev_vect, size*sizeof(floating_type), cudaMemcpyDeviceToHost);
}