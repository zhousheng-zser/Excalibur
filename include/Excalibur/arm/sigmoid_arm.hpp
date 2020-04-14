#ifndef _SIGMOID_ARM_HPP_
#define _SIGMOID_ARM_HPP_

#include "../sigmoid.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class sigmoid_arm : virtual public sigmoid
		{
		public:
			void Forward_cpu(const std::shared_ptr<memory::tensor<float>>& bottom);
		};
	}
}

#endif // _SIGMOID_ARM_HPP_
