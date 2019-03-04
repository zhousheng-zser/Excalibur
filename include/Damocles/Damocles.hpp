#ifndef _DAMOCLES_HPP_
#define _DAMOCLES_HPP_

#include "baseCNN.hpp"

namespace glasssix
{
	namespace longinus
	{
		class Damocles
		{
		public:

			Damocles() {}

			Damocles(int device);

			~Damocles();

			std::vector<FaceInfoX> Detect(const unsigned char* img, const int channels, const int height, const int width,
				const int min_size, const float* threshold, const float factor, const int stage);

		private:
			BaseCNN *baseCNN_;
		};
	}
}

#endif // !_DAMOCLES_HPP_