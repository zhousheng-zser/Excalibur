#include "LABFeatureMap.hpp"
#ifdef USE_CUDA

namespace excalibur
{

	__global__ void ComputeRectSumKernel(const int* int_img, int* rect_sum, int width, int height, int width_, int rect_width_, int rect_height_)
	{
		CUDA_KERNEL_LOOP(index, height)
		{
			const int* top_left = int_img + (index - 0) * width_;
			const int* top_right = top_left + rect_width_ - 1;
			const int* bottom_left = top_left + rect_height_ * width_;
			const int* bottom_right = bottom_left + rect_width_ - 1;
			int* dest = rect_sum + (index + 1) * width_;

			*(dest++) = (*bottom_right) - (*top_right);
			for (int j = 0; j < width; j++)
			{
				dest[j] = (bottom_right + 1)[j] - (top_right + 1)[j] - bottom_left[j] + top_left[j];
			}
		}
	}

	__global__ void ComputeFeatureMapKernel(const int* rect_sum_data, unsigned char* feat_map_data, int width, int height, int offset, int rect_width_, int rect_height_, int width_)
	{
		CUDA_KERNEL_LOOP(index, (width+1) * (height+1))
		{
			int r = index%width;
			int c = index - r*width;
			unsigned char* dest = feat_map_data + r * width_ + c;;
			*dest = 0;
			int white_rect_sum = rect_sum_data[(r + rect_height_) * width_ + c + rect_width_];
			int black_rect_idx = r * width_ + c;

			*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x80 : 0x0);
			black_rect_idx += rect_width_;
			*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x40 : 0x0);
			black_rect_idx += rect_width_;
			*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x20 : 0x0);
			black_rect_idx += offset;
			*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x08 : 0x0);
			black_rect_idx += offset;
			*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x01 : 0x0);
			black_rect_idx -= rect_width_;
			*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x02 : 0x0);
			black_rect_idx -= rect_width_;
			*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x04 : 0x0);
			black_rect_idx -= offset;
			*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x10 : 0x0);
		}
	}

	void LABFeatureMap::ComputeRectSumGPU()
	{
		int width = width_ - rect_width_;
		int height = height_ - rect_height_;
		const int* int_img = int_img_->gpu_data();
		int* rect_sum = rect_sum_->mutable_gpu_data();

		*rect_sum = *(int_img + (rect_height_ - 1) * width_ + rect_width_ - 1);
		MathHelper::VectorSubGPU(int_img + (rect_height_ - 1) * width_ +
			rect_width_, int_img + (rect_height_ - 1) * width_, rect_sum + 1, width);

		ComputeRectSumKernel << <CUDA_GET_BLOCKS(height), CUDA_NUM_THREADS >> >
			(int_img, rect_sum, width, height, width_, rect_width_, rect_height_);
	}

	void LABFeatureMap::ComputeFeatureMapGPU()
	{
		int width = width_ - rect_width_ * num_rect_;
		int height = height_ - rect_height_ * num_rect_;
		int offset = width_ * rect_height_;
		ComputeFeatureMapKernel << <CUDA_GET_BLOCKS((width + 1) * (height + 1)), CUDA_NUM_THREADS >> >
			(rect_sum_data, feat_map_data, width, height, offset, rect_width_, rect_height_, width_);
	}
}

#endif