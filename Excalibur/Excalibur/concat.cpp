#include "concat.hpp"

namespace excalibur
{
	concat::concat(int concat_axis, int device)
	{
		concat_axis_ = concat_axis;
		device_ = device;
	}


	concat::~concat()
	{
	}

	void concat::Forward_cpu(const std::vector<std::shared_ptr<tensor>> bottom, std::shared_ptr<tensor>& top)
	{
		if (bottom.size() <= 1)
		{
			LOG(ERROR) << "One or less input is meaningless.";
			return;
		}
		std::vector<int> top_shape;
		for (int i = 0; i < bottom[0]->data_shape().size(); i++)
		{
			if (i== concat_axis_)
			{
				int counter = 0;
				for (int j = 0; j < bottom.size(); j++)
				{
					counter += bottom[j]->data_shape()[i];
				}
				top_shape.push_back(counter);
			}
			else
			{
				for (int j = 1; j < bottom.size(); j++)
				{
					if (bottom[j]->data_shape()[i]!= bottom[0]->data_shape()[i])
					{
						LOG(ERROR) << "Bottom "<<j<<" has the different data shape in axis "<<i;
						return;
					}
				}
				top_shape.push_back(bottom[0]->data_shape()[i]);
			}
		}
		top.reset(new tensor(top_shape, device_));
		num_concats_ = bottom[0]->count(0, concat_axis_);
		concat_input_size_ = bottom[0]->count(concat_axis_ + 1, bottom[0]->data_shape().size());
		//
		float* top_data = top->mutable_cpu_data();
		int offset_concat_axis = 0;
		const int top_concat_axis = top->data_shape()[concat_axis_];
		for (int i = 0; i < bottom.size(); ++i) {
			const float* bottom_data = bottom[i]->cpu_data();
			const int bottom_concat_axis = bottom[i]->data_shape()[concat_axis_];
			for (int n = 0; n < num_concats_; ++n) {
				math_functions::excalibur_copy(	bottom_concat_axis * concat_input_size_,
					bottom_data + n * bottom_concat_axis * concat_input_size_,
					top_data + (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
					device_);
			}
			offset_concat_axis += bottom_concat_axis;
		}
	}
}
