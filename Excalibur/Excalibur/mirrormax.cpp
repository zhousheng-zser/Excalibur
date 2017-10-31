#include "mirrormax.hpp"
#include <algorithm>

namespace excalibur
{
	mirrormax::mirrormax(int mirror_axis, int device)
	{
		mirror_axis_ = mirror_axis;
		device_ = device;
	}


	mirrormax::~mirrormax()
	{
	}

	void mirrormax::Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		if (mirror_axis_==0)
		{
			int num = bottom->num();
			int channels = bottom->channels();
			int height = bottom->height();
			int width = bottom->width();
			CHECK_EQ(num % 2, 0);
			top.reset(new tensor(std::vector<int>{num / 2, channels, height, width}, device_));
			const float* bottom_data = bottom->cpu_data();
			float* top_data = top->mutable_cpu_data();
			const int mirror_offset = num / 2 * channels * height * width;
			for (int n = 0; n < mirror_offset; n++)
			{
				top_data[n] = std::max(bottom_data[n], bottom_data[n + mirror_offset]);
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}

}
