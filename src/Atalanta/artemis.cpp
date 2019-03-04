#include "artemis.hpp"

namespace excalibur
{
	artemis::artemis()
	{
	}


	artemis::~artemis()
	{
	}

	int artemis::GetDeviceCount()
	{
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		return deviceCount;
	}

	std::string artemis::GetDeviceName(int dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		return std::string(deviceProp.name);
	}


	int artemis::GetDeviceDriverVersion(int dev)
	{
		int driverVersion;
		cudaSetDevice(dev);
		cudaDriverGetVersion(&driverVersion);
		return driverVersion;
	}

	int artemis::GetDeviceRuntimeVersion(int dev)
	{
		int runtimeVersion;
		cudaSetDevice(dev);
		cudaRuntimeGetVersion(&runtimeVersion);
		return runtimeVersion;
	}

	int artemis::GetDeviceCapability(int dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		return deviceProp.major * 10 + deviceProp.minor;
	}

	std::vector<int> artemis::GetDeviceMemory(int dev)
	{
		std::vector<int> output;
		size_t total_byte;
		size_t free_byte;
		cudaSetDevice(dev);
		cudaMemGetInfo(&free_byte, &total_byte);
		output.push_back((int)(total_byte / 1048576.0f));
		output.push_back((int)(free_byte / 1048576.0f));
		return output;
	}

	int artemis::GetDeviceCUDACoreNum(int dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		return _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
	}

}
