#ifndef _INNER_PRODUCT_ARM_HPP
#define _INNER_PRODUCT_ARM_HPP

#include <glasssix/tensor.hpp>
#include "../inner_product.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class inner_product_arm : virtual public inner_product
		{
		public:
			inner_product_arm(std::vector<int> input_shape_withpout_num, int num_output, bool bias_term, int device) : inner_product(input_shape_withpout_num, num_output, bias_term, -1) {}
			void Forward_cpu(std::shared_ptr<tensor<float> >& bottom, std::shared_ptr<tensor<float> >& top);
		};
	}
}

#endif // _INNER_PRODUCT_ARM_HPP
