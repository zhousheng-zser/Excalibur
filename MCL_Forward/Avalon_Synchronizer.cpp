#include "Avalon_Synchronizer.hpp"
#include <memory.h>

namespace Excalibur
{
	template <typename Dtype>
	Avalon_Synchronizer<Dtype>::~Avalon_Synchronizer()
	{
		if (cpu_ptr_ && own_cpu_data_) {
			ExcaliburFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
		}
#ifdef USE_CUDA
		if (gpu_ptr_ && own_gpu_data_) 
		{
			int initial_device;
			cudaGetDevice(&initial_device);
			if (gpu_device_ != -1) 
			{
				CUDA_CHECK(cudaSetDevice(gpu_device_));
			}
			CUDA_CHECK(cudaFree(gpu_ptr_));
			cudaSetDevice(initial_device);
		}
#endif  // USE_CUDA
	}

	template <typename Dtype>
	void Avalon_Synchronizer<Dtype>::to_cpu()
	{
		switch (head_) 
		{
		case UNINITIALIZED:
			ExcaliburMallocHost(&cpu_ptr_, size_*sizeof(Dtype), &cpu_malloc_use_cuda_);
			memset(cpu_ptr_, 0, size_ * sizeof(Dtype));
			head_ = HEAD_AT_CPU;
			own_cpu_data_ = true;
			break;
		case HEAD_AT_GPU:
#ifdef USE_CUDA
			if (cpu_ptr_ == NULL) 
			{
				ExcaliburMallocHost(&cpu_ptr_, size_ * sizeof(Dtype), &cpu_malloc_use_cuda_);
				own_cpu_data_ = true;
			}
			if (gpu_ptr_ != cpu_ptr_)
			{
				CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_ * sizeof(Dtype), cudaMemcpyDefault));
			}
			//excalibur_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
			head_ = SYNCED;
#else
			NO_GPU;
#endif
			break;
		case HEAD_AT_CPU:
		case SYNCED:
			break;
		}
	}

	template <typename Dtype>
	void Avalon_Synchronizer<Dtype>::to_gpu()
	{
#ifdef USE_CUDA
		switch (head_) 
		{
		case UNINITIALIZED:
			CUDA_CHECK(cudaGetDevice(&gpu_device_));
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
			//excalibur_gpu_memset(size_, 0, gpu_ptr_);
			CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_ * sizeof(Dtype)));
			head_ = HEAD_AT_GPU;
			own_gpu_data_ = true;
			break;
		case HEAD_AT_CPU:
			if (gpu_ptr_ == NULL) 
			{
				CUDA_CHECK(cudaGetDevice(&gpu_device_));
				CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
				own_gpu_data_ = true;
			}
			if (cpu_ptr_!=gpu_ptr_)
			{
				CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_*sizeof(Dtype), cudaMemcpyDefault));
			}
			//excalibur_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
			head_ = SYNCED;
			break;
		case HEAD_AT_GPU:
		case SYNCED:
			break;
		}
#else
		NO_GPU;
#endif
	}

	template <typename Dtype>
	void Avalon_Synchronizer<Dtype>::set_cpu_data(void* data)
	{
		CHECK(data);
		if (own_cpu_data_) 
		{
			ExcaliburFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
		}
		cpu_ptr_ = data;
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
	}

	template <typename Dtype>
	void Avalon_Synchronizer<Dtype>::set_cpu_data(void* data, size_t size)
	{
		CHECK(data);
		if (own_cpu_data_)
		{
			ExcaliburFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
		}
		ExcaliburMallocHost(&cpu_ptr_, size * sizeof(Dtype), &cpu_malloc_use_cuda_);
		memcpy(cpu_ptr_, data, size*sizeof(Dtype));
		size_ = size;
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
	}

	template <typename Dtype>
	void Avalon_Synchronizer<Dtype>::set_gpu_data(void* data)
	{
#ifdef USE_CUDA
		CHECK(data);
		if (own_gpu_data_) 
		{
			int initial_device;
			cudaGetDevice(&initial_device);
			if (gpu_device_ != -1) 
			{
				CUDA_CHECK(cudaSetDevice(gpu_device_));
			}
			CUDA_CHECK(cudaFree(gpu_ptr_));
			cudaSetDevice(initial_device);
		}
		gpu_ptr_ = data;
		head_ = HEAD_AT_GPU;
		own_gpu_data_ = false;
#else
		NO_GPU;
#endif
	}

	template <typename Dtype>
	const void* Avalon_Synchronizer<Dtype>::cpu_data()
	{
		to_cpu();
		return (const void*)cpu_ptr_;
	}

	template <typename Dtype>
	const void* Avalon_Synchronizer<Dtype>::gpu_data()
	{
#ifdef USE_CUDA
		to_gpu();
		return (const void*)gpu_ptr_;
#else
		NO_GPU;
		return NULL;
#endif
	}

	template <typename Dtype>
	void* Avalon_Synchronizer<Dtype>::mutable_cpu_data() 
	{
		to_cpu();
		head_ = HEAD_AT_CPU;
		return cpu_ptr_;
	}

	template <typename Dtype>
	void* Avalon_Synchronizer<Dtype>::mutable_gpu_data() 
	{
#ifdef USE_CUDA
		to_gpu();
		head_ = HEAD_AT_GPU;
		return gpu_ptr_;
#else
		NO_GPU;
		return NULL;
#endif
	}

#ifdef USE_CUDA
	template <typename Dtype>
	void Avalon_Synchronizer<Dtype>::async_gpu_push(const cudaStream_t& stream)
	{
		CHECK(head_ == HEAD_AT_CPU);
		if (gpu_ptr_ == NULL) 
		{
			CUDA_CHECK(cudaGetDevice(&gpu_device_));
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
			own_gpu_data_ = true;
		}
		const cudaMemcpyKind put = cudaMemcpyHostToDevice;
		CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
		// Assume caller will synchronize on the stream before use
		head_ = SYNCED;
	}
#endif

	template class Avalon_Synchronizer<float>;
	template class Avalon_Synchronizer<double>;
}