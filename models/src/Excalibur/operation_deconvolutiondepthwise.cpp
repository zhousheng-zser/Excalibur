#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/operation_deconvolutiondepthwise.hpp"
#include "./operation_make_border.hpp"
#include <algorithm>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_deconvolutiondepthwise<Dtype>::operation_deconvolutiondepthwise(const operation_param& param) : operation_deconvolution<Dtype>(param)
		{

		}

		template<typename Dtype>
		void operation_deconvolutiondepthwise<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			operation_deconvolution<Dtype>::forward_cpu_f32(bottoms, tops);
		}

		template<typename Dtype>
		void operation_deconvolutiondepthwise<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			operation_deconvolution<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
				cublas_handle_,
#ifdef USE_CUDNN
				cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
				bottoms, tops);
		}

		INSTANCE_CLASS(operation_deconvolutiondepthwise);
		REGISTE(operation_deconvolutiondepthwise);
	}
}