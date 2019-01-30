#ifdef USE_CUDA
#include "concat.hpp"

namespace glasssix
{
	namespace excalibur
	{
		__global__ void Concat(const int nthreads, const float* in_data,
			const bool forward, const int num_concats, const int concat_size,
			const int top_concat_axis, const int bottom_concat_axis,
			const int offset_concat_axis, float* out_data) {
			CUDA_KERNEL_LOOP(index, nthreads) {
				const int total_concat_size = concat_size * bottom_concat_axis;
				const int concat_num = index / total_concat_size;
				const int concat_index = index % total_concat_size;
				const int top_index = concat_index +
					(concat_num * top_concat_axis + offset_concat_axis) * concat_size;
				if (forward) {
					out_data[top_index] = in_data[index];
				}
				else {
					out_data[index] = in_data[top_index];
				}
			}
		}

		void concat::Forward_native_gpu(const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top)
		{
			if (bottom.size() <= 1)
			{
				LOG(ERROR) << "One or less input is meaningless.";
				return;
			}
			std::vector<int> top_shape;
			for (int i = 0; i < bottom[0]->data_shape().size(); i++)
			{
				if (i == concat_axis_)
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
						if (bottom[j]->data_shape()[i] != bottom[0]->data_shape()[i])
						{
							LOG(ERROR) << "Bottom " << j << " has the different data shape in axis " << i;
							return;
						}
					}
					top_shape.push_back(bottom[0]->data_shape()[i]);
				}
			}
			top.reset(new tensor<float>(top_shape, device_));
			num_concats_ = bottom[0]->count(0, concat_axis_);
			concat_input_size_ = bottom[0]->count(concat_axis_ + 1, bottom[0]->data_shape().size());
			//
			float* top_data = top->mutable_gpu_data();
			int offset_concat_axis = 0;
			const int top_concat_axis = top->data_shape()[concat_axis_];
			const bool kForward = true;
			for (int i = 0; i < bottom.size(); ++i) {
				const float* bottom_data = bottom[i]->gpu_data();
				const int bottom_concat_axis = bottom[i]->data_shape()[concat_axis_];
				const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
				const int nthreads = bottom_concat_size * num_concats_;
				Concat << <CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS >> >(
					nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
					top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
				offset_concat_axis += bottom_concat_axis;
			}
		}
	}
}


#endif