#ifndef _PRELU_ARM_HPP_
#define _PRELU_ARM_HPP_
#include <glasssix/tensor.hpp>
#include "../prelu.hpp"
namespace glasssix
{
	namespace excalibur 
	{
		class prelu_arm : virtual public prelu
		{
		public:
			prelu_arm(int input_channel, bool isrelu = false, int device = -1, bool is_shared = false) : prelu(input_channel, isrelu, -1, is_shared) {}
			void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom);
		};
	}
}

#endif // _PRELU_ARM_HPP_
