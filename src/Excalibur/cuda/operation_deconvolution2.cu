#include "../../../include/Excalibur/operation_deconvolution2.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{

#ifdef USE_CUDA
		template<typename Dtype>
		void operation_deconvolution2<Dtype>::forward_gpu_f32(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			forward_cpu_f32(bottoms, tops);
		}		

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_deconvolution2);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_deconvolution2);
#endif

#endif
	}
}
