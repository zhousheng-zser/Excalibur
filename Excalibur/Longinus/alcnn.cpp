#include "alcnn.hpp"
#include "../Excalibur/io.hpp"

namespace glasssix
{
	alcnn::alcnn(int device)
	{
		device_ = device;
		ipbbox = new ipbbox_net(device_);
		ipts = new ipts_net(device_);
		if (device_>=0)
		{
#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
#else
			NO_GPU;
#endif
		}
	}

	alcnn::~alcnn()
	{
		if (device_ >= 0)
		{
#ifdef USE_CUDA
			if (cublas_handle_)
			{
				CUBLAS_CHECK(cublasDestroy(cublas_handle_));
			}
#else
			NO_GPU;
#endif
		}
		delete ipbbox;
		delete ipts;
	}

	void alcnn::Forward_IPBbox(const std::shared_ptr<tensor> input_data)
	{
		if (device_>=0)
		{
#ifdef USE_CUDA
			ipbbox->Forward_native_gpu(input_data, cublas_handle_);
#else
			NO_GPU;
#endif
		}
		else
		{
			ipbbox->Forward_cpu(input_data);
		}
	}

	void alcnn::Forward_IPTs(const std::shared_ptr<tensor> input_data)
	{
		if (device_ >= 0)
		{
#ifdef USE_CUDA
			ipts->Forward_native_gpu(input_data, cublas_handle_);
#else
			NO_GPU;
#endif
		}
		else
		{
			ipts->Forward_cpu(input_data);
		}
	}
}