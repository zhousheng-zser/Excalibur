#include "slice.hpp"
#ifdef USE_CUDA

namespace excalibur
{
	__global__ void Slice(const int nthreads, const float* in_data,
		const bool forward, const int num_slices, const int slice_size,
		const int bottom_slice_axis, const int top_slice_axis,
		const int offset_slice_axis, float* out_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int total_slice_size = slice_size * top_slice_axis;
			const int slice_num = index / total_slice_size;
			const int slice_index = index % total_slice_size;
			const int bottom_index = slice_index +
				(slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
			if (forward) {
				out_data[index] = in_data[bottom_index];
			}
			else {
				out_data[bottom_index] = in_data[index];
			}
		}
	}

	void slice::Forward_native_gpu(const std::shared_ptr<tensor<float>>& bottom, std::vector<std::shared_ptr<tensor<float>>>& top)
	{
		std::vector<int> top_shape = bottom->data_shape();
		const int bottom_slice_axis = bottom->data_shape()[slice_axis_];
		num_slices_ = bottom->count(0, slice_axis_);
		slice_size_ = bottom->count(slice_axis_ + 1, bottom->data_shape().size());
		int count = 0;
		if (slice_point_.size() == 0)
		{
			top_shape[slice_axis_] = bottom_slice_axis / top.size();
			for (int i = 0; i < top.size(); ++i) {
				top[i].reset(new tensor<float>(top_shape, device_));
				count += top[i]->count(0, top[i]->data_shape().size());
			}
		}
		CHECK_EQ(count, bottom->count(0, bottom->data_shape().size()));
		//
		if (top.size() == 1) { return; }
		int offset_slice_axis = 0;
		const float* bottom_data = bottom->gpu_data();
		const bool kForward = true;
		for (int i = 0; i < top.size(); ++i) {
			float* top_data = top[i]->mutable_gpu_data();
			const int top_slice_axis = top[i]->data_shape()[slice_axis_];
			const int top_slice_size = top_slice_axis * slice_size_;
			const int nthreads = top_slice_size * num_slices_;
			Slice << <CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS >> >(
					nthreads, bottom_data, kForward, num_slices_, slice_size_,
					bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
			offset_slice_axis += top_slice_axis;
		}
	}
}

#endif