#pragma once
#ifndef _SIGMOID_HPP_
#define _SIGMOID_HPP_
#include "Primitives/tensor.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class sigmoid
		{
		public:
			sigmoid();

			virtual ~sigmoid();

			virtual void Forward_cpu(const std::shared_ptr<memory::tensor<float>>& bottom);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>>& bottom);
#endif
		};
	}
}


#endif
