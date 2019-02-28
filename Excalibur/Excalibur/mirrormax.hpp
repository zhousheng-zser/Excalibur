#pragma once
#ifndef _MIRRORMAX_HPP_
#define _MIRRORMAX_HPP_
#include <glasssix\tensor.hpp>
#include "math_functions.hpp"


namespace glasssix
{
	namespace excalibur
	{
		class mirrormax
		{
			int mirror_axis_;
			int device_;
			orderType order_;

		public:
			mirrormax(int mirror_axis, int device);
			~mirrormax();

			void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}


#endif