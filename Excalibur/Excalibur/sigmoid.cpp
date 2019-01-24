#include "sigmoid.hpp"
#include <memory>
#include <algorithm>

namespace excalibur
{
	sigmoid::sigmoid(int input_channel, int device)
	{
		channel_ = input_channel;
		device_ = device;
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
