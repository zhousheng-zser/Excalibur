#include "slice.hpp"

namespace excalibur
{
	slice::slice(int slice_axis, int device)
	{
		slice_point_.clear();
		slice_axis_ = slice_axis;
		device_ = device;
	}

	slice::~slice()
	{
	}

	void slice::Forward_cpu(const tensor* bottom, std::vector<tensor*>& top)
	{
		std::vector<int> top_shape = bottom->data_shape();
		const int bottom_slice_axis = bottom->data_shape()[slice_axis_];
		num_slices_ = bottom->count(0, slice_axis_);
		slice_size_ = bottom->count(slice_axis_ + 1, bottom->data_shape().size());
		int count = 0;
		if (slice_point_.size()==0)
		{
			top_shape[slice_axis_] = bottom_slice_axis / top.size();
			for (int i = 0; i < top.size(); ++i) {
				top[i] = new tensor(top_shape, device_);
				count += top[i]->count(0, top[i]->data_shape().size());
			}
		}
		CHECK_EQ(count, bottom->count(0, bottom->data_shape().size()));
		//
		if (top.size()==1)
		{
			return;
		}
		int offset_slice_axis = 0;
		const float* bottom_data = bottom->cpu_data();
		//const int bottom_slice_axis = bottom->data_shape()[slice_axis_];
		for (int i = 0; i < top.size(); ++i)
		{
			float* top_data = top[i]->mutable_cpu_data();
			const int top_slice_axis = top[i]->data_shape()[slice_axis_];
			for (int n = 0; n < num_slices_; ++n)
			{
				const int top_offset = n * top_slice_axis * slice_size_;
				const int bottom_offset =
					(n * bottom_slice_axis + offset_slice_axis) * slice_size_;
				memcpy(top_data + top_offset, bottom_data + bottom_offset, top_slice_axis * slice_size_ * sizeof(float));
			}
			offset_slice_axis += top_slice_axis;
		}
	}

}
