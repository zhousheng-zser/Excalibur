#pragma once
#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_
#include "Primitives/tensor.hpp"
#include "math_functions.hpp"
#ifdef USE_CUDNN
#include "cudnn.hpp"
#endif

namespace glasssix
{
	namespace excalibur
	{
		class softmax
		{
		protected:
			int outer_num_;
			int inner_num_;
			int softmax_axis_;

			/// sum_multiplier is used to carry out sum using BLAS
			std::shared_ptr<memory::tensor<float>> sum_multiplier_;
			/// scale is an intermediate Blob to hold temporary results.
			std::shared_ptr<memory::tensor<float>> scale_;

			int device_;
			memory::orderType order_;

		public:
			softmax(int input_channel, int device);
			virtual ~softmax();

			virtual void Forward_cpu(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top);
#ifdef USE_CUDNN
		private:
			float one = 1.0, zero = 0.0;
			cudnnHandle_t cudnn_handle_ = nullptr;
			cudnnTensorDescriptor_t bottom_desc_;
			cudnnTensorDescriptor_t top_desc_;
		public:
			void Forward_gpu_cudnn(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top);
#endif
#endif
		};
	}
}


#endif