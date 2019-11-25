#ifndef _POOLING_ARM_HPP_
#define _POOLING_ARM_HPP_

#include "../pooling.hpp"

namespace glasssix {
	namespace excalibur
	{
		class pooling_arm : virtual public pooling
		{
		public:
			pooling_arm(int kernel, int stride, int pad, int type, int device) : pooling(kernel, stride, pad, type, -1) {}
			void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
		};
	}
}

#endif // _POOLING_ARM_HPP_
