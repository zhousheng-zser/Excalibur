#pragma once
#ifndef _RESHAPE_HPP_
#define _RESHAPE_HPP_
#include "../../include/Primitives/tensor.hpp"
#include <memory>

namespace glasssix
{
	namespace excalibur
	{
		class reshape
		{
		protected:
			std::vector<int> shape_param_;
			int channel_;
			int height_;
			int width_;
			int device_;
			memory::orderType order_;

		public:
			reshape(int dimension1, int dimension2, int dimension3, int dimension4, int device = -1);

			virtual ~reshape();

			virtual void Forward(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top);

		};
	}
}


#endif