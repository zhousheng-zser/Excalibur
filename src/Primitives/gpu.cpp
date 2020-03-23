#include "../../include/Primitives/gpu.hpp"

namespace glasssix
{
#ifdef x86
#ifdef USE_CUDA
#if CUDART_VERSION < 5000
#include <cuda.h>
	// This function wraps the CUDA Driver API into a template function
	template <class T>
	inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
	{
		CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);
		if (CUDA_SUCCESS != error)
		{
			fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
				error, __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}
	}
#endif /* CUDART_VERSION < 5000 */

	int get_cuda_device_count()
	{
		int deviceCount = 0;
		CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
		return deviceCount;
	}

	std::string get_cuda_device_name(int dev)
	{
		CUDA_CHECK(cudaSetDevice(dev));
		cudaDeviceProp deviceProp;
		CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
		return std::string(deviceProp.name);
	}

	int get_cuda_device_driver_version(int dev)
	{
		int driverVersion;
		CUDA_CHECK(cudaSetDevice(dev));
		CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
		return driverVersion;
	}

	int get_cuda_device_runtime_version(int dev)
	{
		int runtimeVersion;
		CUDA_CHECK(cudaSetDevice(dev));
		CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
		return runtimeVersion;
	}

	int get_cuda_device_capability(int dev)
	{
		CUDA_CHECK(cudaSetDevice(dev));
		cudaDeviceProp deviceProp;
		CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
		return deviceProp.major * 10 + deviceProp.minor;
	}

	void get_cuda_device_memory(int dev, std::size_t &total_size, std::size_t &free_size)
	{
		CUDA_CHECK(cudaSetDevice(dev));
		CUDA_CHECK(cudaMemGetInfo(&free_size, &total_size));
		total_size /= 1048576.0f;
		free_size /= 1048576.0f;
	}

	int get_cuda_device_cuda_core_num(int dev)
	{
		/// TODO: _ConvertSMVer2Cores should be re-write
		//CUDA_CHECK(cudaSetDevice(dev));
		//cudaDeviceProp deviceProp;
		//cudaGetDeviceProperties(&deviceProp, dev);
		//return _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
		return -1;
	}
#endif //!USE_CUDA
#endif //!x86
}