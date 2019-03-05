#ifndef _BASE_CNN_HPP_
#define _BASE_CNN_HPP_

#include <vector>
#include "../Longinus/common.hpp"
namespace glasssix
{
	namespace longinus
	{
		class vDamocles
		{
		public:

			vDamocles() {}

			vDamocles(int device){}

			virtual ~vDamocles() {}

			virtual std::vector<FaceInfomation> Detect(const unsigned char* img, const int channels, const int height, const int width,
				const int min_size, const float* threshold, const float factor, const int stage) = 0;
		};
	}
}
#endif //!_BASE_CNN_HPP_