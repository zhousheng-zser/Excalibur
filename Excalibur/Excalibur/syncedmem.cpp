#include "syncedmem.hpp"

namespace excalibur
{
	syncedmem::syncedmem() 
		: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
		own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false), device_(-1){
#ifdef USE_CUDA
#ifdef _DEBUG
		if (device_ >= 0) {
			CUDA_CHECK(cudaGetDevice(&device_));
		}
#endif
#endif
	}

	syncedmem::syncedmem(size_t size)
		: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
		own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false), device_(-1) {
#ifdef USE_CUDA
#ifdef _DEBUG
		if (device_ >= 0) {
			CUDA_CHECK(cudaGetDevice(&device_));
		}
#endif
#endif
	}


	syncedmem::syncedmem(size_t size, int device)
		: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
		own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false), device_(device){
#ifdef USE_CUDA
#ifdef _DEBUG
		if (device_>=0){
			CUDA_CHECK(cudaGetDevice(&device_));
		}
#endif
#endif
	}


	syncedmem::~syncedmem()
	{
		//check_device();
		if (cpu_ptr_ && own_cpu_data_) {
			FreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
		}

#ifdef USE_CUDA
		if (gpu_ptr_ && own_gpu_data_) {
#ifdef FAST_ALLOCATION
			gpu_free(gpu_ptr_);
#else
			CUDA_CHECK(cudaFree(gpu_ptr_));
#endif
		}
#endif  // CPU_ONLY
	}

	void syncedmem::to_cpu()
	{
		//check_device();
		switch (head_) {
		case UNINITIALIZED:
			MallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, device_);
			memset(cpu_ptr_, 0, size_);
			head_ = HEAD_AT_CPU;
			own_cpu_data_ = true;
			break;
		case HEAD_AT_GPU:
#ifdef USE_CUDA
			if (cpu_ptr_ == NULL) {
				MallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, device_);
				own_cpu_data_ = true;
			}
			if (gpu_ptr_!= cpu_ptr_)
			{
				CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDefault));
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


	void syncedmem::to_gpu()
	{
		//check_device();
#ifdef USE_CUDA
		switch (head_) {
		case UNINITIALIZED:
#ifdef FAST_ALLOCATION
			gpu_ptr_ = gpu_malloc(size_);
#else
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
#endif
			cudaMemset(gpu_ptr_, 0, size_);
			head_ = HEAD_AT_GPU;
			own_gpu_data_ = true;
			break;
		case HEAD_AT_CPU:
			if (gpu_ptr_ == NULL) {
#ifdef FAST_ALLOCATION
				gpu_ptr_ = gpu_malloc(size_);
#else
				CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
#endif
				own_gpu_data_ = true;
			}
			CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyDefault));
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

	void syncedmem::check_device()
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

	const void* syncedmem::cpu_data()
	{
		//check_device();
		to_cpu();
		return (const void*)cpu_ptr_;
	}

	const void* syncedmem::gpu_data()
	{
		//check_device();
#ifdef USE_CUDA
		to_gpu();
		return (const void*)gpu_ptr_;
#else
		NO_GPU;
		return NULL;
#endif
	}

	void syncedmem::set_cpu_data(void* data)
	{
		//check_device();
		CHECK(data);
		if (own_cpu_data_) {
			FreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
		}
		cpu_ptr_ = data;
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
	}

	void syncedmem::set_gpu_data(void* data)
	{
		check_device();
#ifdef USE_CUDA
		CHECK(data);
		if (own_gpu_data_) {
#ifdef FAST_ALLOCATION
			gpu_free(gpu_ptr_);
#else
			CUDA_CHECK(cudaFree(gpu_ptr_));
#endif
		}
		gpu_ptr_ = data;
		head_ = HEAD_AT_GPU;
		own_gpu_data_ = false;
#else
		NO_GPU;
#endif
	}

	void* syncedmem::mutable_cpu_data()
	{
		//check_device();
		to_cpu();
		head_ = HEAD_AT_CPU;
		return cpu_ptr_;
	}

	void* syncedmem::mutable_gpu_data()
	{
		check_device();
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
	void syncedmem::async_gpu_push(const cudaStream_t& stream) {
		check_device();
		CHECK(head_ == HEAD_AT_CPU);
		if (gpu_ptr_ == NULL) {
#ifdef FAST_ALLOCATION
			gpu_ptr_ = gpu_malloc(size_);
#else
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
#endif
			own_gpu_data_ = true;
		}
		const cudaMemcpyKind put = cudaMemcpyHostToDevice;
		CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
		// Assume caller will synchronize on the stream before use
		head_ = SYNCED;
	}
#endif
}
