#pragma once
#ifndef _ARTEMIS_HPP_
#define _ARTEMIS_HPP_

#include "../Excalibur/accelerator.hpp"
#include <memory>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>


namespace excalibur
{
#if CUDART_VERSION < 5000

	// CUDA-C includes
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

	class artemis
	{
	public:
		artemis();
		~artemis();
		static int GetDeviceCount();
		static std::string GetDeviceName(int dev);
		static int GetDeviceDriverVersion(int dev);
		static int GetDeviceRuntimeVersion(int dev);
		static int GetDeviceCapability(int dev);
		static std::vector<int> GetDeviceMemory(int dev);
		static int GetDeviceCUDACoreNum(int dev);
	};
}

#endif