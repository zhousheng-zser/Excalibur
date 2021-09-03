#include "gpu_common.hpp"

namespace glasssix
{

    std::uint32_t lwp_id()
	{
#if defined(APPLE) || defined(WIN32)
        // return static_cast<std::uint32_t>(std::this_thread::get_id());
        auto tid = std::this_thread::get_id();
        _Thrd_t t = *(_Thrd_t*)(char*)&tid;
        return t._Id;
#else
        return static_cast<std::uint32_t>(syscall(SYS_gettid));
#endif
    }

    std::uint64_t lwp_dev_id(int dev)
	{
        std::uint64_t dev64 = static_cast<std::uint64_t>(dev < 0 ? cudaGetDevice(&dev) : dev);
        return lwp_id() + (dev64 << 32LL);
    }

#ifdef USE_CUDA
    CUDAStream::CUDAStream(bool high_priority)
	{
        if (high_priority) 
        {
            int leastPriority, greatestPriority;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
            CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, greatestPriority));
        }
        else 
        {
            CUDA_CHECK(cudaStreamCreate(&stream_));
        }
        int dev = 0;
        cudaGetDevice(&dev);
        DLOG(INFO) << "New " << (high_priority ? "high priority " : "") << "stream "
            << stream_ << ", device " << dev << ", thread "
            << lwp_id();
    }

    CUDAStream::~CUDAStream()
	{
        int current_device;  // Just to check CUDA status:
        cudaError_t status = cudaGetDevice(&current_device);
        // Preventing dead lock while Excalibur shutting down.
        if (status != cudaErrorCudartUnloading) 
        {
            CUDA_CHECK(cudaStreamDestroy(stream_));
        }
    }

    CUBLASHandle::CUBLASHandle(std::shared_ptr<CUDAStream> stream)
        : handle_(nullptr), stream_(std::move(stream))
	{
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetStream(handle_, stream_->get()));
    }
	
    CUBLASHandle::~CUBLASHandle()
	{
        CUBLAS_CHECK(cublasDestroy(handle_));
    }
#ifdef USE_CUDNN
    CUDNNHandle::CUDNNHandle(std::shared_ptr<CUDAStream> stream)
        : handle_(nullptr), stream_(std::move(stream))
	{
        CUDNN_CHECK(cudnnCreate(&handle_));
        CUDNN_CHECK(cudnnSetStream(handle_, stream_->get()));
    }
	
    CUDNNHandle::~CUDNNHandle()
	{
        CUDNN_CHECK(cudnnDestroy(handle_));
    }
#endif
#endif
}
