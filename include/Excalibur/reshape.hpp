#pragma once
#ifndef _RESHAPE_HPP_
#define _RESHAPE_HPP_
#include <glasssix/tensor.hpp>
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
			orderType order_;

		public:
			reshape(int dim1, int dim2, int dim3, int dim4);

			virtual ~reshape();

			virtual void Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);

		};
	}
}


#endif