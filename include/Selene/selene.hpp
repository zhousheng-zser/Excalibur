#ifndef _SELENE_HPP_
#define _SELENE_HPP_

#include <memory>
#include <vector>
#include <iostream>

namespace glasssix
{
	namespace longinus
	{
		class Selene
		{
		public:

			Selene() {}

			Selene(int device){}

			virtual ~Selene() {}

			virtual bool judge(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) = 0;
		};
	}
}
#endif //!_BASE_SELENE_HPP_