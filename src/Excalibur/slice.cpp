#include "slice.hpp"

using namespace glasssix::memory;

namespace glasssix
{
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

		void slice::Forward_cpu(const std::shared_ptr<tensor<float>> bottom, std::vector<std::shared_ptr<tensor<float>>>& top)
		{
			std::vector<int> top_shape = bottom->data_shape();
			const int bottom_slice_axis = bottom->data_shape()[slice_axis_];
			num_slices_ = bottom->count(0, slice_axis_);
			slice_size_ = bottom->count(slice_axis_ + 1, bottom->data_shape().size());
			int count = 0;
			if (slice_point_.size() == 0)
			{
				top_shape[slice_axis_] = bottom_slice_axis / top.size();
				int top_num = top.size();
				top.clear();
				for (int i = 0; i < top_num; ++i) {
					std::shared_ptr<tensor<float>> temp = nullptr;
					temp.reset(new tensor<float>(top_shape, device_));
					top.push_back(temp);
					//top[i].reset(new tensor(top_shape, device_));
					count += top[i]->count(0, top[i]->data_shape().size());
				}
			}
			CHECK_EQ(count, bottom->count(0, bottom->data_shape().size()));
			//
			if (top.size() == 1)
			{
				return;
			}
			int offset_slice_axis = 0;
			const float* bottom_data = bottom->cpu_data();
			for (int i = 0; i < top.size(); ++i)
			{
				float* top_data = top[i]->mutable_cpu_data();
				const int top_slice_axis = top[i]->data_shape()[slice_axis_];
				for (int n = 0; n < num_slices_; ++n)
				{
					const int top_offset = n * top_slice_axis * slice_size_;
					const int bottom_offset =
						(n * bottom_slice_axis + offset_slice_axis) * slice_size_;
					math_functions::excalibur_copy(top_slice_axis * slice_size_ * sizeof(float),
						bottom_data + bottom_offset, top_data + top_offset, device_);
				}
				offset_slice_axis += top_slice_axis;
			}
		}


		void slice::Forward_cpu(const std::shared_ptr<tensor<float>> bottom, std::shared_ptr<tensor<float>>& top1, std::shared_ptr<tensor<float>>& top2)
		{
			std::vector<int> top_shape = bottom->data_shape();
			const int bottom_slice_axis = bottom->data_shape()[slice_axis_];
			num_slices_ = bottom->count(0, slice_axis_);
			slice_size_ = bottom->count(slice_axis_ + 1, bottom->data_shape().size());
			int count = 0;
			top_shape[slice_axis_] = bottom_slice_axis / 2;
			top1.reset(new tensor<float>(top_shape, device_));
			top2.reset(new tensor<float>(top_shape, device_));
			int offset_slice_axis = 0;
			const float* bottom_data = bottom->cpu_data();
			//
			float* top_data = top1->mutable_cpu_data();
			const int top_slice_axis = top1->data_shape()[slice_axis_];
			for (int n = 0; n < num_slices_; ++n)
			{
				const int top_offset = n * top_slice_axis * slice_size_;
				const int bottom_offset =
					(n * bottom_slice_axis + offset_slice_axis) * slice_size_;
				math_functions::excalibur_copy(top_slice_axis * slice_size_ * sizeof(float),
					bottom_data + bottom_offset, top_data + top_offset, device_);
			}
			offset_slice_axis += top_slice_axis;
			//
			top_data = top2->mutable_cpu_data();
			//top_slice_axis = top1->data_shape()[slice_axis_];
			for (int n = 0; n < num_slices_; ++n)
			{
				const int top_offset = n * top_slice_axis * slice_size_;
				const int bottom_offset =
					(n * bottom_slice_axis + offset_slice_axis) * slice_size_;
				math_functions::excalibur_copy(top_slice_axis * slice_size_ * sizeof(float),
					bottom_data + bottom_offset, top_data + top_offset, device_);
			}
			offset_slice_axis += top_slice_axis;
		}

	}
}

