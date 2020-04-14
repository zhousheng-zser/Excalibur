#ifndef _ELTWISE_ARM_HPP_
#define _ELTWISE_ARM_HPP_

#include <vector>
#include <memory>
#include "../../Primitives/tensor.hpp"
#include "../eltwise.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class eltwise_arm : virtual public eltwise
		{
		public:
			eltwise_arm(int type, int device) : eltwise(type, -1) {}
			void Forward_cpu(const std::vector<std::shared_ptr<memory::tensor<float>>> bottom, std::shared_ptr<memory::tensor<float>>& top);
		};
	}
}

#endif