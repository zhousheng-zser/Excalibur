#include "../../include/Excalibur/operation_shape.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_shape<Dtype>::operation_shape(const operation_param& param) : operation<Dtype>(param)
		{

		}

		template<typename Dtype>
		void operation_shape<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			auto shape = bottoms[0]->data_shape();
			tops[0].reset(new memory::tensor<float>(static_cast<int>(shape.size()), -1, memory::NCHW, nullptr));
			auto top_data = tops[0]->mutable_cpu_data();
			for (size_t i = 0; i < shape.size(); i++)
				top_data[i] = shape[i];
		}

#ifdef USE_CUDA
		template<typename Dtype>
		void operation_shape<Dtype>::forward_gpu_f32(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			forward_cpu_f32(bottoms, tops);
		}
#endif //!USE_CUDA

#ifdef USE_CUDA
		template<typename Dtype>
		void operation_shape<Dtype>::forward_gpu_f16(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			auto shape = bottoms[0]->data_shape();
			tops[0].reset(new memory::tensor<unsigned short>(static_cast<int>(shape.size()), -1, memory::NCHW, nullptr));
			auto top_data = tops[0]->mutable_cpu_data();
			for (size_t i = 0; i < shape.size(); i++)
				top_data[i] = static_cast<unsigned short>(shape[i]);
		}
#endif //!USE_CUDA

		INSTANCE_CLASS(operation_shape);
		REGISTE(operation_shape);
	}
}