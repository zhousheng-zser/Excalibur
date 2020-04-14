#include "../../include/Excalibur/hswish.hpp"

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		hswish::hswish() {}

		hswish::~hswish() {}

		void hswish::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
		{
			float* bottom_data = (bottom)->mutable_cpu_data();
			size_t count = bottom->count();
			for (size_t i = 0; i < count; i++)
			{
				if ((bottom_data[i] + 3) >=0 && (bottom_data[i] + 3) <= 6)
				{
					bottom_data[i] = bottom_data[i] * (bottom_data[i] + 3) / 6;
				}
				else
				{
					bottom_data[i] = 0.0f;
				}
			}
		}
	}
}