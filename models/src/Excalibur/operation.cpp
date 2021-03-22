#include "../../include/Excalibur/operation.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template <>
		void operation<float>::forward_cpu(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			if (params_.int8_quantization_)
			{
				forward_cpu_i8(bottoms, tops);
			}
			else
			{
				forward_cpu_f32(bottoms, tops);
			}
		}

		template <>
		void operation<float>::forward_gpu(
#ifdef USE_CUDA
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			if (params_.int8_quantization_)
			{
				forward_gpu_i8(
#ifdef USE_CUDA
					cublas_handle_,
#ifdef USE_CUDNN
					cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
					bottoms, tops);
			}
			else
			{
				forward_gpu_f32(
#ifdef USE_CUDA
					cublas_handle_,
#ifdef USE_CUDNN
					cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
					bottoms, tops);
			}
		}

		template <>
		void operation<unsigned short>::forward_cpu(const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops)
		{
			forward_cpu_f16(bottoms, tops);
		}

		template <>
		void operation<unsigned short>::forward_gpu(
#ifdef USE_CUDA
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops)
		{
			forward_gpu_f16(
#ifdef USE_CUDA
				cublas_handle_,
#ifdef USE_CUDNN
				cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
				bottoms, tops);
		}

		template <>
		void operation<double>::forward_cpu(const std::vector<std::shared_ptr<memory::tensor<double>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<double>>>& tops)
		{
			forward_cpu_d64(bottoms, tops);
		}

		template <>
		void operation<double>::forward_gpu(
#ifdef USE_CUDA
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<double>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<double>>>& tops)
		{
			forward_gpu_d64(
#ifdef USE_CUDA
				cublas_handle_,
#ifdef USE_CUDNN
				cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
				bottoms, tops);
		}


		// instantiate class
		template class operation<float>;
		template class operation<double>;
		template class operation<unsigned short>;
	}
}