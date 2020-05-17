/*!
 * \file   linear_equations.c
 * \brief  A gaussian elimination solver.
 * \author (C) Copyright 2018 by Peter C. Chapin <pchapin@vtc.edu>
 *
 * This is the serial version of the algorithm.
 */

#include <math.h>
#include <string.h>
#include <pthread.h>
#include <CL/cl.h>
#include <sys/sysinfo.h>
#include <stdbool.h>
#include <stdio.h>
#include "linear_equations.h"

///////////////////////////////////////////////////////////////////////

typedef struct Params
{
  int row_start;
  int row_end;
  int matrix_size;
  int operation_row;
  floating_type* matrix;
  floating_type* vector;
} Params;

///////////////////////////////////////////////////////////////////////

// Globals

pthread_barrier_t step_barrier;
pthread_barrier_t swap_barrier;


///////////////////////////////////////////////////////////////////////

inline void subtract_multiples(const int first_row,
                               const int last_row,
                               const int size,
                               const int operation_row,
                               floating_type* matrix,
                               floating_type* vector)
{
  static bool normal_direction = true;

  floating_type m;
  floating_type* a = matrix;
  floating_type* b = vector;
  const int i = operation_row;
  const bool switch_directions = false;
  const int actual_first = (first_row > (operation_row + 1)) ? first_row : (operation_row + 1);

  const int row_start = (switch_directions && !normal_direction) ? last_row : actual_first;
  const int row_end = (switch_directions && !normal_direction) ? actual_first : last_row;

  // Subtract multiples of row i from subsequent rows.
  for( int j = row_start;
       normal_direction ? j <= row_end : j >= row_end;
       normal_direction ? j++ : j-- )
  {
    m = MATRIX_GET( a, size, j, i ) / MATRIX_GET( a, size, i, i );

    for( int k = 0; k < size; ++k )
    {
      MATRIX_PUT(a,
                 size,
                 j,
                 k,
                 MATRIX_GET( a, size, j, k ) -
                       m * MATRIX_GET( a, size, i, k ) );
    }
    b[j] -= m * b[i];
  }

  if(switch_directions)
  {
    normal_direction = !normal_direction;
  }
}


///////////////////////////////////////////////////////////////////////

//! Does the elimination step of reducing the system.
static int elimination( int size, floating_type *a, floating_type *b )
{
    floating_type *temp_array = (floating_type *)malloc( size * sizeof(floating_type) );
    int            i, j, k;
    floating_type  temp, m;

    for( i = 0; i < size - 1; ++i ) {

        // Find the row with the largest value of |a[j][i]|, j = i, ..., n - 1
        k = i;
        m = fabs( MATRIX_GET( a, size, i, i ) );
        for( j = i + 1; j < size; ++j ) {
            if( fabs( MATRIX_GET( a, size, j, i ) ) > m ) {
                k = j;
                m = fabs( MATRIX_GET( a, size, j, i ) );
            }
        }

        // Check for |a[k][i]| zero.
        if( fabs( MATRIX_GET( a, size, k, i ) ) <= 1.0E-6 ) {
            return -2;
        }

        // Exchange row i and row k, if necessary.
        if( k != i ) {
            memcpy( temp_array, MATRIX_GET_ROW( a, size, i ), size * sizeof( floating_type ) );
            memcpy( MATRIX_GET_ROW( a, size, i ), MATRIX_GET_ROW( a, size, k ), size * sizeof( floating_type ) );
            memcpy( MATRIX_GET_ROW( a, size, k ), temp_array, size * sizeof( floating_type ) );
            
            // Exchange corresponding elements of b.
            temp = b[i];
            b[i] = b[k];
            b[k] = temp;
        }

        subtract_multiples(i+1,
                           size-1,
                           size,
                           i,
                           a,
                           b);
    }
    return 0;
}


//! Does the back substitution step of solving the system.
static int back_substitution( int size, floating_type *a, floating_type *b )
{
    floating_type sum;
    int           i, j;

    for( i = size - 1; i >=0; --i ) {
        if( fabs( MATRIX_GET( a, size, i, i ) ) <= 1.0E-6 ) {
            return -2;
        }

        sum = b[i];
        for( j = i + 1; j < size; ++j ) {
            sum -= MATRIX_GET( a, size, i, j ) * b[j];
        }
        b[i] = sum / MATRIX_GET( a, size, i, i );
    }
    return 0;
}


int gaussian_solve( int size, floating_type *a, floating_type *b )
{
    int return_code = elimination( size, a, b );
    if( return_code == 0 )
        return_code = back_substitution( size, a, b );
    return return_code;
}

///////////////////////////////////////////////////////////////////////

int gaussian_solve_threaded( int size, 
                             floating_type *a,
                             floating_type *b,
                             ThreadPool* pool,
                             enum Thread_type type )
{
  int return_code;
  if(type == EType_serial)
  {
    return_code = elimination(size, a, b);
  }
  else if((type == EType_pthread) || (type == EType_pool))
  {
   return_code = elimination_thread(size, a, b, pool, type );
  }
  else if(type == EType_barrier)
  {
    return_code = elimination_barriers(size, a, b);
  }
  else if(type == EType_cuda)
  {
    return_code = elimination_cuda(size, a, b);
  }
  else if(type == EType_opencl)
  {
    return_code = elimination_opencl(size, a, b);
  }

  if( return_code == 0 )
  {
    return_code = back_substitution( size, a, b );
  }
  return return_code;
}

///////////////////////////////////////////////////////////////////////

/**
 * @brief eliminate_row
 * @param arg
 * @return
 */
void* eliminate_rows(void* arg)
{
  const struct Params* param = (struct Params*)arg;
  subtract_multiples(param->row_start,
                     param->row_end,
                     param->matrix_size,
                     param->operation_row,
                     param->matrix,
                     param->vector);

  return NULL;
}

///////////////////////////////////////////////////////////////////////

/**
 * @brief Do gausian elimination using multiple threads
 * @param size of the input
 * @param a matrix of the inputs
 * @param b vector of the outputs
 * @return error code. 0 on sucsess, negative on failure
 */
int elimination_thread(int size, 
                       floating_type *a, 
                       floating_type *b, 
                       ThreadPool* pool,
                       enum Thread_type thr_type)
{
  floating_type *temp_array =
      (floating_type *)malloc( size * sizeof(floating_type) );
  int i, j, k, thread_rows, row_start, pthreads_threads_running, pool_threads_running;
  floating_type temp, m;

  const int nproc = get_nprocs();
  pthread_t thread_ids_pthread[nproc];
  struct Params param_array[nproc];
  threadid_t thread_ids_pools[nproc];

  for( i = 0; i < size - 1; ++i )
  {
    // Find the row with the largest value of |a[j][i]|, j = i, ..., n - 1
    k = i;
    m = fabs( MATRIX_GET( a, size, i, i ) );
    for( j = i + 1; j < size; ++j )
    {
      if( fabs( MATRIX_GET( a, size, j, i ) ) > m )
      {
        k = j;
        m = fabs( MATRIX_GET( a, size, j, i ) );
      }
    }

    // Check for |a[k][i]| zero.
    if( fabs( MATRIX_GET( a, size, k, i ) ) <= 1.0E-6 )
    {
      return -2;
    }

    // Exchange row i and row k, if necessary.
    if( k != i )
    {
      memcpy(temp_array,
             MATRIX_GET_ROW( a, size, i ),
             size * sizeof( floating_type ) );
      memcpy(MATRIX_GET_ROW( a, size, i ),
             MATRIX_GET_ROW( a, size, k ),
             size * sizeof( floating_type ) );
      memcpy(MATRIX_GET_ROW( a, size, k ),
             temp_array,
             size * sizeof( floating_type ) );

      // Exchange corresponding elements of b.
      temp = b[i];
      b[i] = b[k];
      b[k] = temp;
    }

    // Max number of rows we are giving to each thread
    // TODO: Could have a min number of rows here (to start less threads)
    thread_rows = (size - (i+1)) / nproc;
    row_start = i + 1;

    pthreads_threads_running = 0;
    pool_threads_running = 0;
    for(int t = 0; t < nproc; t++)
    {
      int row_end = row_start + thread_rows;
      if(row_end >= size)
      {
        if(row_start >= size)
        {
          break; // No need to start this thread
        }

        row_end = size - 1;
      }

      param_array[t].row_start = row_start;
      param_array[t].row_end = row_end;
      param_array[t].matrix_size = size;
      param_array[t].operation_row = i;
      param_array[t].matrix = a;
      param_array[t].vector = b;

      if(thr_type == EType_pthread)
      {
        pthread_create(&thread_ids_pthread[t], NULL, eliminate_rows, &param_array[t]);
        pthreads_threads_running++;
      }
      else if(thr_type == EType_pool)
      {
        thread_ids_pools[t] = ThreadPool_start(pool, eliminate_rows, &param_array[t]);
        pool_threads_running++;
      }
      
      row_start = row_end + 1;
    }

    for(int t = 0; t < pthreads_threads_running; t++)
    {
      pthread_join(thread_ids_pthread[t], NULL);
    }

    for(int t = 0; t < pool_threads_running; t++)
    {
      ThreadPool_result(pool, thread_ids_pools[t]);
    }
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////

void* run_barriers(void* arg)
{
  const struct Params* param = (struct Params*)arg;
  const int size = param->matrix_size;
  floating_type *temp_array =
      (floating_type *)malloc( size * sizeof(floating_type) );
  int i, j, k;
  floating_type temp, m;
  floating_type* a = param->matrix;
  floating_type* b = param->vector;

  for( i = 0; i < size - 1; ++i )
  {
    if( pthread_barrier_wait( &step_barrier ) == PTHREAD_BARRIER_SERIAL_THREAD )
    {
      // Find the row with the largest value of |a[j][i]|, j = i, ..., n - 1
      k = i;
      m = fabs( MATRIX_GET( a, size, i, i ) );
      for( j = i + 1; j < size; ++j )
      {
        if( fabs( MATRIX_GET( a, size, j, i ) ) > m )
        {
          k = j;
          m = fabs( MATRIX_GET( a, size, j, i ) );
        }
      }

      // Check for |a[k][i]| zero.
      if( fabs( MATRIX_GET( a, size, k, i ) ) <= 1.0E-6 )
      {
        return -2;
      }

      // Exchange row i and row k, if necessary.
      if( k != i )
      {
        memcpy(temp_array,
               MATRIX_GET_ROW( a, size, i ),
               size * sizeof( floating_type ) );
        memcpy(MATRIX_GET_ROW( a, size, i ),
               MATRIX_GET_ROW( a, size, k ),
               size * sizeof( floating_type ) );
        memcpy(MATRIX_GET_ROW( a, size, k ),
               temp_array,
               size * sizeof( floating_type ) );

        // Exchange corresponding elements of b.
        temp = b[i];
        b[i] = b[k];
        b[k] = temp;
      }
    }

    pthread_barrier_wait(&swap_barrier);

    subtract_multiples(param->row_start,
                       param->row_end,
                       param->matrix_size,
                       i,
                       param->matrix,
                       param->vector);
  }
}

///////////////////////////////////////////////////////////////////////

int elimination_barriers(int size, 
                         floating_type *a, 
                         floating_type *b )
{
  const int nproc = get_nprocs();
  const int thread_rows = (size - 1) / nproc;
  int row_start = 1;
  
  pthread_t thread_ids[nproc];
  struct Params param_array[nproc];

  pthread_barrier_init( &step_barrier, NULL, nproc );
  pthread_barrier_init( &swap_barrier, NULL, nproc );

  for(int t = 0; t < nproc; t++)
  {
    int row_end = row_start + thread_rows;
    if(row_end >= size)
    {
      row_end = size - 1;
    }

    param_array[t].row_start = row_start;
    param_array[t].row_end = row_end;
    param_array[t].matrix_size = size;
    param_array[t].operation_row = 0; // All threads will go through all rows
    param_array[t].matrix = a;
    param_array[t].vector = b;

    row_start = row_end + 1;

    pthread_create(&thread_ids[t], NULL, run_barriers, &param_array[t]);
  }

  // Wait for all thread to end.
  for( int i = 0; i < nproc; ++i ) 
  {
    pthread_join( thread_ids[i], NULL );
  }

  pthread_barrier_destroy( &step_barrier );
  pthread_barrier_destroy( &swap_barrier );

  return 0;
}

///////////////////////////////////////////////////////////////////////

int elimination_cuda(int size,
                     floating_type* a,
                     floating_type* b)
{

}

///////////////////////////////////////////////////////////////////////

// TODO: Subtract multiples

const char *programSource =
    "__kernel\n"
    "void sub_mult(__global double* A,\n"
    "              __global double* B)\n"
    "{\n"
    "    int idx = get_global_id( 0 );\n"
    "    C[idx] = 1;\n" // TODO: Do subtract multiples
    "}\n";

///////////////////////////////////////////////////////////////////////

int elimination_opencl(int size,
                       floating_type* a,
                       floating_type* b)
{
    floating_type *temp_array =
      (floating_type *)malloc( size * sizeof(floating_type) );
  int i, j, k, thread_rows, row_start, pthreads_threads_running, pool_threads_running;
  floating_type temp, m;

  for( i = 0; i < size - 1; ++i )
  {
    // Find the row with the largest value of |a[j][i]|, j = i, ..., n - 1
    k = i;
    m = fabs( MATRIX_GET( a, size, i, i ) );
    for( j = i + 1; j < size; ++j )
    {
      if( fabs( MATRIX_GET( a, size, j, i ) ) > m )
      {
        k = j;
        m = fabs( MATRIX_GET( a, size, j, i ) );
      }
    }

    // Check for |a[k][i]| zero.
    if( fabs( MATRIX_GET( a, size, k, i ) ) <= 1.0E-6 )
    {
      return -2;
    }

    // Exchange row i and row k, if necessary.
    if( k != i )
    {
      memcpy(temp_array,
             MATRIX_GET_ROW( a, size, i ),
             size * sizeof( floating_type ) );
      memcpy(MATRIX_GET_ROW( a, size, i ),
             MATRIX_GET_ROW( a, size, k ),
             size * sizeof( floating_type ) );
      memcpy(MATRIX_GET_ROW( a, size, k ),
             temp_array,
             size * sizeof( floating_type ) );

      // Exchange corresponding elements of b.
      temp = b[i];
      b[i] = b[k];
      b[k] = temp;
    }

    // TODO: Would like to do a lot of this setup once

    const int mat_datasize = size * size * sizeof(floating_type);
    const int vect_datasize = size * sizeof(floating_type);
  
    cl_uint platform_count = 0;
    cl_platform_id platform;
  
    // Get the first platform.
    cl_int status = clGetPlatformIDs(1, &platform, &platform_count);
    printf( "%d OpenCL platforms available.\n", (int)platform_count );
    if( status != CL_SUCCESS ) fprintf( stderr, "Error getting platform IDs!\n" );
  
    // Get the first device.
    cl_device_id device;
    status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL );
    if( status != CL_SUCCESS ) fprintf( stderr, "Error getting device IDs!\n" );
  
    // Create a context and associate it with the device.
    cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
    if( status != CL_SUCCESS ) fprintf( stderr, "Error creating context!\n" );
  
    // Create a command queue and associate it with the device.
    cl_command_queue cmdQueue = clCreateCommandQueue( context, device, 0, &status );
    if( status != CL_SUCCESS ) fprintf( stderr, "Error creating the command queue!\n" );
  
    // Allocate two input buffers and one output buffer.
    cl_mem bufA = clCreateBuffer( context, CL_MEM_READ_WRITE, mat_datasize, NULL, &status );
    cl_mem bufB = clCreateBuffer( context, CL_MEM_READ_WRITE, vect_datasize, NULL, &status );
  
    // Transfer data from host arrays into the input buffers.
    status = clEnqueueWriteBuffer( cmdQueue, bufA, CL_FALSE, 0, mat_datasize, a, 0, NULL, NULL );
    status = clEnqueueWriteBuffer( cmdQueue, bufB, CL_FALSE, 0, vect_datasize, b, 0, NULL, NULL );
  
    // Create a program with source code.
    cl_program program =
        clCreateProgramWithSource( context, 1, (const char **)&programSource, NULL, &status );
    if( status != CL_SUCCESS ) fprintf( stderr, "Error creating program object!\n" );
  
    // Build the program for the device.
    status = clBuildProgram( program, 1, &device, NULL, NULL, NULL );
    if( status != CL_SUCCESS ) {
        switch( status ) {
        case CL_COMPILER_NOT_AVAILABLE:
            fprintf( stderr, "Error building program! Compiler not available.\n" );
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            fprintf( stderr, "Error building program! Build failure.\n" );
            break;
        default:
            fprintf( stderr, "Error building program! *unknown reason*\n" );
            break;
        }
    }
  
    // Create the vector addition kernel.
    cl_kernel kernel = clCreateKernel( program, "vecadd", &status );
    if( status != CL_SUCCESS ) fprintf( stderr, "Error creating kernel!\n" );
  
    // Set the kernel arguments.
    status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &bufA );
    status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &bufB );
  
    // Define an index space of work items for execution.
    // A work group size is not required, but can be used.
    size_t indexSpaceSize[1], workGroupSize[1];
    indexSpaceSize[0] = size;
    workGroupSize[0] = 256;
  
    // Execute the kernel.
    status =
        clEnqueueNDRangeKernel(
          cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);
    if( status != CL_SUCCESS ) fprintf( stderr, "Error queuing kernel execution!\n" );
  
    // Read the device output buffer to the host output arrays.
    status = clEnqueueReadBuffer( cmdQueue, bufA, CL_TRUE, 0, mat_datasize, a, 0, NULL, NULL );
    status = clEnqueueReadBuffer( cmdQueue, bufB, CL_TRUE, 0, vect_datasize, b, 0, NULL, NULL );

  
    // Free OpenCL resources.
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufA );
    clReleaseMemObject( bufB );
    clReleaseContext( context );
  }


}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
