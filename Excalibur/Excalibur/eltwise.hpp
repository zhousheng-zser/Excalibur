#pragma once
#ifndef _ELTWISE_HPP_
#define _ELTWISE_HPP_
#include <glasssix\tensor.hpp>
#include "math_functions.hpp"


namespace glasssix
{
	namespace excalibur
	{
		class eltwise
		{
			enum eltwise_type { SUM, MAX };
			eltwise_type type_;
			std::vector<float>coeffs_;
			int device_;
		public:
			eltwise(int type, int device);

			~eltwise();

			void Forward_cpu(const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top);

#ifdef USE_CUDA
			void Forward_native_gpu(cublasHandle_t cublas_handle_, const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}


#endif