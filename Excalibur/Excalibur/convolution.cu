#include "convolution.hpp"
#include <stdio.h>  
#include <stdlib.h>  
#include <time.h>  
#include <cuda_runtime.h>  

#define BLOCK_SIZE 16  
static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))  

//#define HANDLE_NULL( a ) {if ((a) == NULL) { \  
//printf("Host memory failed in %s at line %d\n", \
//	__FILE__, __LINE__); \
//	exit(EXIT_FAILURE); }}

static void GenerateNumbers(int *number, int size)
{
	for (int i = 0; i < size; i++)
	{
		number[i] = rand() % 10;
	}
}

static bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for (i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}

	if (i >= count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

//1个block，block内256个thread，threadIdx.x = 0的线程计时，每个线程计算一个结果  
//最终的结果由threadIdx.x = 0的线程进行累加，临时结果存放在块内共享内存中  
__global__ static void sumOfSquares(int *num, int size, int* result, clock_t* time)
{
	extern __shared__ int temp[];
	int sum = 0;
	clock_t start;
	const int tid = threadIdx.x;
	temp[tid] = 0;

	if (tid == 0)
	{
		start = clock();
	}

	for (int index = tid; index < size; index += blockDim.x)
	{
		sum += num[index] * num[index];
	}

	temp[tid] = sum;
	__syncthreads();

	if (tid == 0)
	{
		sum = 0;
		for (int index = 0; index < blockDim.x; index++)
		{
			sum += temp[index];
		}

		*result = sum;
		*time = clock() - start;
	}
}