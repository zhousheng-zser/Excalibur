#include "../../include/Excalibur/operation_split.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_split<Dtype>::operation_split(const operation_param& param) : operation<Dtype>(param)
		{
			
		}

		template<typename Dtype>
		void operation_split<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			for (size_t i = 0; i < tops.size(); i++)
			{
				tops[i] = bottoms[0];
			}
		}


		template<typename Dtype>
		void operation_split<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			NOT_IMPLEMENTED;
		}

		INSTANCE_CLASS(operation_split);
		REGISTE(operation_split);
	}
}