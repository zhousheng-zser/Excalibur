#ifdef USE_CUDA
#include "math_functions.hpp"
#include "eltwise.hpp"

namespace glasssix
{
	namespace excalibur
	{
		__global__ void MaxForward(const int nthreads, const float* bottom_data_a,
			const float* bottom_data_b, const int blob_idx,
			float* top_data) {
			CUDA_KERNEL_LOOP(index, nthreads) {
				top_data[index] = max(bottom_data_a[index], bottom_data_b[index]);
			}
		}

		void eltwise::Forward_gpu_native(cublasHandle_t &cublas_handle_, const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top)
		{
			coeffs_ = std::vector<float>(bottom.size(), 1);
			top.reset(new tensor<float>(bottom[0]->data_shape(), device_, bottom[0]->order()));
			const int count = top->count();
			float* top_data = top->mutable_gpu_data();
			switch (type_)
			{
			case SUM:
				math_functions::gpu_set(count, 0.0f, top_data);
				for (int i = 0; i < bottom.size(); ++i)
				{
					math_functions::gpu_saxpy(cublas_handle_, count, coeffs_[i], bottom[i]->gpu_data(), top_data);
				}
				break;
			case MAX:
				MaxForward << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
					count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data);
				for (int i = 2; i < bottom.size(); ++i) {
					// NOLINT_NEXT_LINE(whitespace/operators)
					MaxForward << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
						count, top_data, bottom[i]->gpu_data(), i - 1, top_data);
				}
				break;
			default:
				LOG(FATAL) << "Unknown elementwise operation.";
			}
		}
	}
}


#endif