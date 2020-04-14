#ifndef _SOFTMAX_ARM_HPP_
#define _SOFTMAX_ARM_HPP_

#include "../softmax.hpp"

namespace glasssix 
{
	namespace excalibur
	{
		class softmax_arm : virtual public softmax
		{
		public:
			softmax_arm(int input_channel, int device) : softmax(input_channel, -1) {}
			void Forward_cpu(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top);
		};
	}
}

#endif // _SOFTMAX_ARM_HPP_
