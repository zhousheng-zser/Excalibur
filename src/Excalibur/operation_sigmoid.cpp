#include "../../include/Excalibur/operation_sigmoid.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template <typename Dtype>
		operation_sigmoid<Dtype>::operation_sigmoid(const operation_param &param) : operation<Dtype>(param)
		{
			this->params_.inplace_ = true;
		}

		template <typename Dtype>
		void operation_sigmoid<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
													   std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				float *top_data = tops[i]->mutable_cpu_data();
				const float *bottom_data = bottoms[i]->cpu_data();
				const int count = bottoms[i]->count();
				for (int i = 0; i < count; ++i)
				{
					top_data[i] = 1. / (1. + exp(-bottom_data[i]));
				}
			}
		}

		// 		template<typename Dtype>
		// 		void operation_sigmoid<Dtype>::forward_gpu_f32(
		// #ifdef USE_CUDA
		// 			cublasHandle_t &cublas_handle_,
		// #ifdef USE_CUDNN
		// 			cudnnHandle_t cudnn_handle,
		// #endif //!USE_CUDNN
		// #endif //!USE_CUDA
		// 			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
		// 			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		// 		{
		// 			forward_cpu_f32(bottoms, tops);
		// 		}

#ifndef USE_CUDA
		STUB_GPU(operation_slice);
#endif

		INSTANCE_CLASS(operation_sigmoid);
		REGISTE(operation_sigmoid);
	}
}