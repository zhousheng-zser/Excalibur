#include "sigmoid.hpp"
#include <algorithm>
#include <cmath>

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		sigmoid::sigmoid()
		{
		}

		sigmoid::~sigmoid()
		{

		}


		void sigmoid::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
		{
			float* bottom_data = (bottom)->mutable_cpu_data();
			const int count = bottom->count();
			for (int i = 0; i < count; ++i)
			{
				bottom_data[i] = 1. / (1. + exp(-bottom_data[i]));
			}
		}
	}
}

