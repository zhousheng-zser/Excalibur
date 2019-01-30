#ifndef _BASENET_HPP_
#define _BASENET_HPP_

#include <vector>

namespace glasssix
{
	namespace cassius
	{
		class BaseNet
		{
		public:

			BaseNet() {}

			BaseNet(int device) {}

			virtual ~BaseNet() {}

			virtual std::vector<std::vector<float> > Forward(const float* input_data, unsigned num) = 0;

			virtual std::vector<std::vector<float> > Forward(const unsigned char* input_data, unsigned num) = 0;

		};
	}
}

#endif // !_BASENET_HPP_