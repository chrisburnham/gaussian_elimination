#ifndef CUDA_ELIM_H
#define CUDA_ELIM_H

#include "matrix_defs.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_subtract_multiples(const int size,
							 							 floating_type* matrix,
							 							 floating_type* vector);

#ifdef __cplusplus
}
#endif

#endif