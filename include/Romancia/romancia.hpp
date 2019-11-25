#ifndef _ROMANCIA_HPP_
#define _ROMANCIA_HPP_

#include <vector>
#include <iostream>
#include <memory>

namespace glasssix
{
	namespace longinus
	{
		class Banshee;

		class Romancia
		{
		public:

			Romancia() {}

			Romancia(int device);

			~Romancia();

			void Forward(const float* input_data, int num, int order);

			void Forward(const unsigned char* input_data, int num, int order);

			void getParam(std::vector<std::vector<float> > &keypointParam, int num);

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, 
				std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks);

		private:
			std::shared_ptr<Banshee> bansheelia_;
		};
	}
}

#endif // !_ROMANCIA_HPP_