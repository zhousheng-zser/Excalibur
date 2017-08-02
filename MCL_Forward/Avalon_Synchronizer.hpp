#pragma once
#ifndef _AVALON_SYNCHRONIZER_HPP_
#define _AVALON_SYNCHRONIZER_HPP_
#include "Accelerator.hpp"
#include <cstdlib>

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Excalibur: check mode."

namespace Excalibur
{
	template <typename Dtype>
	class Avalon_Synchronizer
	{
		// If CUDA is available and in GPU mode, host memory will be allocated pinned,
		// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
		// The improvement in performance seems negligible in the single GPU case,
		// but might be more significant for parallel training. Most importantly,
		// it improved stability for large models on many GPUs.
		void ExcaliburMallocHost(void** ptr, size_t size, bool* use_cuda) 
		{
			//mkl_malloc function is thread safe function which enables safe execution by multiple threads at the same time. Such buffer can be 
			//allocated in one thread but freed inanother one. So, it is not TLSdata. In comparison with standard malloc,mkl_malloc allocates memory 
			//from heap but with needed alignment to get better performance via vectorization instructions oncomputation with these data if any. As to 
			//the last argument MKL default value is 64 byte alignment now and corresponds to AVX architecture not to 32/64 bit OS/builds.
#ifdef USE_MKL
			*ptr = mkl_malloc(size ? size : 1, 64);
#else
			*ptr = malloc(size);
#endif
#ifdef USE_CUDA
			if (mode == GPU) 
			{
				CUDA_CHECK(cudaMallocHost(ptr, size));
				*use_cuda = true;
				return;
			}
#endif
			*use_cuda = false;
			CHECK(*ptr) << "host allocation of size " << size << " failed";
		}

		void ExcaliburFreeHost(void* ptr, bool use_cuda) 
		{
#ifdef USE_MKL
			mkl_free(ptr);
#else
			free(ptr);
#endif
#ifdef USE_CUDA
			if (use_cuda) 
			{
				CUDA_CHECK(cudaFreeHost(ptr));
				return;
			}
#endif
		}

	public:
		Avalon_Synchronizer()
			: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
			own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
			gpu_device_(-1), mode(CPU){}
		Avalon_Synchronizer(size_t size, int gpu_device, Avalon mode_)
			: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
			own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false)
		{
			if (mode_==
				CPU
#ifndef x86		
				||ARM
#endif
				)
			{
				gpu_device_ = -1;
				mode = mode_;
			}
#ifdef USE_CUDA
			if (mode_ == GPU)
			{
				gpu_device_ = gpu_device;
				mode = mode_;
			}
#endif
		}
		~Avalon_Synchronizer();
		const void* cpu_data();
		void set_cpu_data(void* data);
		void set_cpu_data(void* data, size_t size);
		const void* gpu_data();
		void set_gpu_data(void* data);
		void* mutable_cpu_data();
		void* mutable_gpu_data();
		enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
		SyncedHead head() { return head_; }
		size_t size() { return size_; }
#ifdef USE_CUDA
		void async_gpu_push(const cudaStream_t& stream);
#endif
	private:
		
		void to_cpu();

		void to_gpu();

		void* cpu_ptr_;
		void* gpu_ptr_;
		size_t size_;
		SyncedHead head_;
		bool own_cpu_data_;
		bool cpu_malloc_use_cuda_;
		bool own_gpu_data_;
		int gpu_device_;
		Avalon mode;
	};
}
#endif //_AVALON_SYNCHRONIZER_HPP_