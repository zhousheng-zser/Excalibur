#include "syncedmem.hpp"
/*
* If CUDA is available and in GPU mode, host memory will be allocated pinned,
* using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
* The improvement in performance seems negligible in the single GPU case,
* but might be more significant for parallel training. Most importantly,
* it improved stability for large models on many GPUs.
*/

namespace glasssix
{
	namespace memory
	{
		template <typename Dtype>
		syncedmem<Dtype>::syncedmem() : cpu_ptr_(nullptr), gpu_ptr_(nullptr), allocator_(nullptr), size_(0), 
			head_(UNINITIALIZED), own_cpu_data_(false), own_gpu_data_(false), device_(-1) 
		{
			if (device_ >= 0)
			{
#ifdef USE_CUDA
				CUDA_CHECK(cudaSetDevice(device_));
#ifdef _DEBUG
				CUDA_CHECK(cudaGetDevice(&device_));
#endif
			}
#else
				NO_GPU;
		}
#endif
		}

		template <typename Dtype>
		syncedmem<Dtype>::syncedmem(size_t size, int device)
		{
			cpu_ptr_ = nullptr;
			gpu_ptr_ = nullptr;
			allocator_ = nullptr;
			size_ = size;
			head_ = UNINITIALIZED;
			own_cpu_data_ = false;
			own_gpu_data_ = false;
			device_ = device;
			if (device_ >= 0)
			{
#ifdef USE_CUDA
				CUDA_CHECK(cudaSetDevice(device_));
#ifdef _DEBUG
				CUDA_CHECK(cudaGetDevice(&device_));
#endif
#else
				NO_GPU;
#endif
			}
		}

		template <typename Dtype>
		syncedmem<Dtype>::~syncedmem()
		{
			if (cpu_ptr_ && own_cpu_data_)
			{
				//aligned_heap_free(cpu_ptr_, cpu_malloc_use_cuda_);
				if (allocator_)
				{
					allocator_->fastFree(cpu_ptr_, device_);
				}
				else
				{
					if (device_ >= 0)
					{
#ifdef USE_CUDA
						CUDA_CHECK(cudaFreeHost(cpu_ptr_));
#else
						NO_GPU;
#endif
					}
					else
					{
						aligned_heap_free(cpu_ptr_);
					}
				}
			}

#ifdef USE_CUDA
			if (gpu_ptr_ && own_gpu_data_)
			{
				CUDA_CHECK(cudaFree(gpu_ptr_));
			}
#endif  // USE_CUDA
		}

		template<typename Dtype>
		void syncedmem<Dtype>::set_allocator(pool_allocator<Dtype>* allocator)
		{
			if (!allocator_)
			{
				allocator_ = allocator;
			}
		}

		template <typename Dtype>
		const Dtype* syncedmem<Dtype>::cpu_data()
		{
			to_cpu();
			return static_cast<const Dtype*>(cpu_ptr_);
		}

		template <typename Dtype>
		const Dtype* syncedmem<Dtype>::gpu_data()
		{
#ifdef USE_CUDA
			to_gpu();
			return static_cast<const Dtype*>(gpu_ptr_);
#else
			NO_GPU;
			return nullptr;
#endif
		}

		template <typename Dtype>
		Dtype* syncedmem<Dtype>::mutable_cpu_data()
		{
			to_cpu();
			head_ = HEAD_AT_CPU;
			return cpu_ptr_;
		}

		template <typename Dtype>
		Dtype* syncedmem<Dtype>::mutable_gpu_data()
		{
#ifdef USE_CUDA
			to_gpu();
			head_ = HEAD_AT_GPU;
			return gpu_ptr_;
#else
			NO_GPU;
			return nullptr;
#endif
		}

#ifdef USE_CUDA
		template <typename Dtype>
		void syncedmem<Dtype>::async_gpu_push(const cudaStream_t& stream)
		{
			check_device();
			CHECK(head_ == HEAD_AT_CPU);
			if (gpu_ptr_ == nullptr) 
			{
				CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_ * sizeof(Dtype)));
				own_gpu_data_ = true;
			}
			const cudaMemcpyKind put = cudaMemcpyHostToDevice;
			CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_ * sizeof(Dtype), put, stream));
			// Assume caller will synchronize on the stream before use
			head_ = SYNCED;
		}
#endif

		template <typename Dtype>
		void syncedmem<Dtype>::check_device()
		{
#ifdef USE_CUDA
#ifdef _DEBUG
			int device;
			cudaGetDevice(&device);
			CHECK(device == device_);
			if (gpu_ptr_ && own_gpu_data_) {
				cudaPointerAttributes attributes;
				CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
				CHECK(attributes.device == device_);
			}
#endif
#endif
		}

		template <typename Dtype>
		void syncedmem<Dtype>::to_cpu()
		{
			switch (head_)
			{
			case UNINITIALIZED:
				//MallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, device_);
				if (allocator_)
				{
					cpu_ptr_ = static_cast<Dtype*>(allocator_->fastMalloc(size_ * sizeof(Dtype), device_));
				}
				else
				{
					if (device_ >= 0)
					{
#ifdef USE_CUDA
						CUDA_CHECK(cudaSetDevice(device_));
						CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_ * sizeof(Dtype)));
#else
						NO_GPU;
#endif
					}
					else
					{
						cpu_ptr_ = static_cast<Dtype*>(aligned_heap_alloc(size_ * sizeof(Dtype)));
					}
				}
				memset(cpu_ptr_, 0, size_ * sizeof(Dtype));
				head_ = HEAD_AT_CPU;
				own_cpu_data_ = true;
				break;
			case HEAD_AT_GPU:
#ifdef USE_CUDA
				if (cpu_ptr_ == nullptr)
				{
					//MallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, device_);
					if (allocator_)
					{
						cpu_ptr_ = static_cast<Dtype*>(allocator_->fastMalloc(size_ * sizeof(Dtype), device_));
					}
					else
					{
						if (device_ >= 0)
						{
#ifdef USE_CUDA
							CUDA_CHECK(cudaSetDevice(device_));
							CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_ * sizeof(Dtype)));
#else
							NO_GPU;
#endif
						}
						else
						{
							cpu_ptr_ = static_cast<Dtype*>(aligned_heap_alloc(size_ * sizeof(Dtype)));
						}
					}
					own_cpu_data_ = true;
				}
				if (gpu_ptr_ != cpu_ptr_)
				{
					CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_ * sizeof(Dtype), cudaMemcpyDefault));
				}
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
		void syncedmem<Dtype>::to_gpu()
		{
#ifdef USE_CUDA
			switch (head_)
			{
			case UNINITIALIZED:
				CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_ * sizeof(Dtype)));
				cudaMemset(gpu_ptr_, 0, size_ * sizeof(Dtype));
				head_ = HEAD_AT_GPU;
				own_gpu_data_ = true;
				break;
			case HEAD_AT_CPU:
				if (gpu_ptr_ == nullptr)
				{
					CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_ * sizeof(Dtype)));
					own_gpu_data_ = true;
				}
				CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_ * sizeof(Dtype), cudaMemcpyDefault));
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

		// instantiate class
		template class syncedmem<float>;
		template class syncedmem<double>;
		template class syncedmem<int>;
		template class syncedmem<int const*>;
		template class syncedmem<unsigned char>;
		template class syncedmem<char>;
		template class syncedmem<signed char>;
		template class syncedmem<short>;
		template class syncedmem<unsigned short>;
		template class syncedmem<unsigned int>;
	}
}
