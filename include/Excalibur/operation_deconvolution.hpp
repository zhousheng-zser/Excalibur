#pragma once
#ifndef _OPERATION_DECONVOLUTION_HPP_
#define _OPERATION_DECONVOLUTION_HPP_

#include "operation_general_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class operation_deconvolution : public operation_general_conv<Dtype>
		{
		public:
			operation_deconvolution(const operation_param& param);

			virtual int init_weights();

			virtual int init_weights(FILE* fp);

		protected:
            void cut_padding(std::shared_ptr<memory::tensor<float>> &top_blob_bordered, std::shared_ptr<memory::tensor<float>> &top_blob);
        
			virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

			virtual void forward_gpu_f32(
#ifdef USE_CUDA
				cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
				cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
				const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);
		};
	}
}

#endif //!_OPERATION_DECONVOLUTION_HPP_