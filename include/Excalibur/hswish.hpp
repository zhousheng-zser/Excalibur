#pragma once
#ifndef _HSWISH_HPP_
#define _HSWISH_HPP_

#include "Primitives/tensor.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class hswish
		{
		public:
			hswish();

			virtual ~hswish();

			virtual void Forward_cpu(const std::shared_ptr<memory::tensor<float>>& bottom);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>>& bottom);
#endif
		};
	}
}
#endif // !_HSWISH_HPP_
