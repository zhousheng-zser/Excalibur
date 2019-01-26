#ifndef _BASENET_HPP_
#define _BASENET_HPP_

#include <vector>

namespace glasssix
{
	namespace longinus
	{
		class BaseNet
		{
		public:

			BaseNet() {}

			BaseNet(int device) {}

			virtual ~BaseNet() {}

			virtual void Forward(const float* input_data, unsigned num) = 0;

			virtual void Forward(const unsigned char* input_data, unsigned num) = 0;

			virtual void getParam(std::vector<std::vector<float> > &keypointParam, unsigned num) = 0;

		};
	}
}

#endif // !_BASENET_HPP_