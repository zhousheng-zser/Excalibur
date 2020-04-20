#ifdef USE_CUDA
#include "excalibur/pca.hpp"

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		void pca::Forward_gpu_native(cublasHandle_t &cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			int num = bottom->num();
			top.reset(new tensor<float>(std::vector<int>{num, final_dimensions}, device_));
			const float* bottom_data = bottom->gpu_data();
			float* top_data = top->mutable_gpu_data();
			math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans,
				num, final_dimensions, initial_dimensions, 1.0f, bottom_data, top_data, 0.0f, top_data);
		}
	}
}


#endif