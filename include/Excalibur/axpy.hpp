#pragma once
#ifndef _AXPY_HPP_
#define _AXPY_HPP_

#include "Primitives/tensor.hpp"
#include "math_functions.hpp"


namespace glasssix
{
	namespace excalibur
	{
		class axpy
		{
			int device_;
			memory::orderType order_;
			std::shared_ptr<memory::tensor<float>> scales_;
			std::shared_ptr<memory::tensor<float>> input_;

		public:
			axpy(int device);

			~axpy();

			void Forward_cpu(const std::vector<std::shared_ptr<memory::tensor<float>>> bottom, std::shared_ptr<memory::tensor<float>>& top);

#ifdef USE_CUDA
			void Forward_gpu_native(cublasHandle_t &cublas_handle_, const std::vector<std::shared_ptr<memory::tensor<float>>> bottom, std::shared_ptr<memory::tensor<float>>& top);
#endif
		};
	}
}


#endif