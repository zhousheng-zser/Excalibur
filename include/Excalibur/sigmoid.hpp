#pragma once
#ifndef _SIGMOID_HPP_
#define _SIGMOID_HPP_
#include <glasssix/tensor.hpp>
//#include <memory>

namespace glasssix
{
	namespace excalibur
	{
		class sigmoid
		{
		public:
			sigmoid();

			~sigmoid();

			void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom);
#endif
		};
	}
}


#endif
