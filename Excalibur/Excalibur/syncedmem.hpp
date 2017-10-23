#pragma once
#ifndef _SYNCEDMEM_HPP_
#define _SYNCEDMEM_HPP_
#include "accelerator.hpp"

namespace excalibur
{
	// If CUDA is available and in GPU mode, host memory will be allocated pinned,
	// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
	// The improvement in performance seems negligible in the single GPU case,
	// but might be more significant for parallel training. Most importantly,
	// it improved stability for large models on many GPUs.
	inline void MallocHost(void** ptr, size_t size, bool* use_cuda, int device_id) {
#ifdef USE_CUDA
		if (device_id >=0) {
			cudaSetDevice(device_id);
			CUDA_CHECK(cudaMallocHost(ptr, size));
			*use_cuda = true;
			return;
		}
#endif
#ifdef USE_MKL
		*ptr = mkl_malloc(size ? size : 1, 64);
#else
		*ptr = malloc(size);
#endif
		*use_cuda = false;
		CHECK(*ptr) << "host allocation of size " << size << " failed";
	}

	inline void FreeHost(void* ptr, bool use_cuda) {
#ifdef USE_CUDA
		if (use_cuda) {
			CUDA_CHECK(cudaFreeHost(ptr));
			return;
		}
#endif
#ifdef USE_MKL
		mkl_free(ptr);
#else
		free(ptr);
#endif
	}

	class syncedmem
	{
	public:
		syncedmem();
		explicit syncedmem(size_t size);
		explicit syncedmem(size_t size, int device);
		~syncedmem();
		enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
		SyncedHead head() { return head_; }
		size_t size() { return size_; }
		const void* cpu_data();
		void set_cpu_data(void* data);
		const void* gpu_data();
		void set_gpu_data(void* data);
		void* mutable_cpu_data();
		void* mutable_gpu_data();
		
#ifdef USE_CUDA
		void async_gpu_push(const cudaStream_t& stream);
#endif
	private:
		void check_device();

		void to_cpu();
		void to_gpu();
		void* cpu_ptr_;
		void* gpu_ptr_;
		size_t size_;
		SyncedHead head_;
		bool own_cpu_data_;
		bool cpu_malloc_use_cuda_;
		bool own_gpu_data_;
		int device_;
	};
}

#endif //_SYNCEDMEM_HPP_