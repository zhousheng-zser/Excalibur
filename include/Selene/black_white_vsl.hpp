#ifndef _BLACK_WHITE_VSL_HPP_
#define _BLACK_WHITE_VSL_HPP_

#include "selene.hpp"
#include "../Excalibur/support_layers.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class Black_white_vsl:public Selene
		{
		public:
			Black_white_vsl(int device);
			virtual ~Black_white_vsl();

			bool judge(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) override;
		};
	}
}

#endif