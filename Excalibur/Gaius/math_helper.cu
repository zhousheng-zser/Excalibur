#include "math_helper.hpp"
#ifdef USE_CUDA

namespace excalibur
{
	__global__ void UInt8ToInt32Kernel(const int nthreads, const unsigned char* src, int* dest)
	{
		CUDA_KERNEL_LOOP(index, nthreads)
		{
			dest[index] = static_cast<int>(src[index]);
		}
	}

	__global__ void VectorAbsKernel(const int nthreads, const int* src, int* dest)
	{
		CUDA_KERNEL_LOOP(index, nthreads)
		{
			dest[index] = (src[index] >= 0 ? src[index] : -src[index]);
		}
	}

	__global__ void VectorAddKernel (const int nthreads, const int* x, const int* y, int* z)
	{
		CUDA_KERNEL_LOOP(index, nthreads)
		{
			z[index] = x[index] + y[index];
		}
	}

	__global__ void VectorSubKernel(const int nthreads, const int* x, const int* y, int* z)
	{
		CUDA_KERNEL_LOOP(index, nthreads)
		{
			z[index] = x[index] - y[index];
		}
	}

	__global__ void SquareKernel(const int nthreads, const int* src, int* dest)
	{
		CUDA_KERNEL_LOOP(index, nthreads)
		{
			dest[index] = src[index] * src[index];
		}
	}

	void MathHelper::UInt8ToInt32GPU(const unsigned char* src, int* dest, int len)
	{
		UInt8ToInt32Kernel << <CUDA_GET_BLOCKS(len), CUDA_NUM_THREADS >> >
			(len, src, dest);
	}

	void MathHelper::VectorAbsGPU(const int* src, int* dest, int len)
	{
		VectorAbsKernel << <CUDA_GET_BLOCKS(len), CUDA_NUM_THREADS >> >
			(len, src, dest);
	}

	void MathHelper::VectorAddGPU(const int* x, const int* y, int* z, int len)
	{
		VectorAddKernel << <CUDA_GET_BLOCKS(len), CUDA_NUM_THREADS >> >
			(len, x, y, z);
	}

	void MathHelper::VectorSubGPU(const int* x, const int* y, int* z, int len)
	{
		VectorSubKernel << <CUDA_GET_BLOCKS(len), CUDA_NUM_THREADS >> >
			(len, x, y, z);
	}

	void MathHelper::SquareGPU(const int* src, int* dest, int len)
	{
		SquareKernel << <CUDA_GET_BLOCKS(len), CUDA_NUM_THREADS >> >
			(len, src, dest);
	}
}

#endif // USE_CUDA
