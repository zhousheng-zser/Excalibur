#ifndef _ROMANCIA_HPP_
#define _ROMANCIA_HPP_

#include "baseNet.hpp"



namespace glasssix
{
	namespace longinus
	{
		class Romancia
		{
		public:

			Romancia() {}

			Romancia(int device);

			~Romancia();

			void Forward(const float* input_data, unsigned num, int order);

			void Forward(const unsigned char* input_data, unsigned num, int order);

			void getParam(std::vector<std::vector<float> > &keypointParam, unsigned num);

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks);

		private:
			BaseNet *baseNet_;
		};
	}
}

#endif // !_ROMANCIA_HPP_