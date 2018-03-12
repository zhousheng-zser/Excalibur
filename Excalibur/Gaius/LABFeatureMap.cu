#include "LABFeatureMap.hpp"

#ifdef USE_CUDA

namespace excalibur
{
	__global__ void SSARectSumKernel(int* dest, const int* bottom_right, const int* top_right, const int* bottom_left, const int* top_left, int width)
	{
		CUDA_KERNEL_LOOP(index, width)
		{
			dest[index] = (bottom_right + 1)[index] - (top_right + 1)[index] - bottom_left[index] + top_left[index];
		}
	}

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
			/*SSARectSumKernel << <(width+ 512 -1) / 512, 512 >> >
				(dest, bottom_right, top_right, bottom_left, top_left, width);*/
		}
	}

	__global__ void ComputeFeatureMapKernel(const int* rect_sum_data, unsigned char* feat_map_data, int width, int height, int offset, int rect_width_, int rect_height_, int width_)
	{
		CUDA_KERNEL_LOOP(index, (width+1) * (height+1))
		{
			int r = index/width;
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

	__global__ void fix_rect_sum(int* rect_sum, const int* int_img, int rect_width_, int rect_height_, int width_)
	{
		*rect_sum = *(int_img + (rect_height_ - 1) * width_ + rect_width_ - 1);
	}

	__global__ void IntegralGPUKernel(const int* src, int* dest, const int* dest_above, int width_, int height_)
	{
		//CUDA_KERNEL_LOOP(index, height_)
		for (int index = 0; index < height_; index++)
		{
			if (index == 0)
			{
				for (int i = 1; i < width_; i++)
				{
					dest[i] = src[i] + dest[i - 1];
				}
			}
			else
			{
				int offset = index * width_;
				int s = 0;
				for (int i = 0; i < width_; i++)
				{
					s += src[offset + i];
					dest[offset + i] = dest_above[offset - width_ + i] + s;
				}
			}
		}
	}

	__global__ void IntegralGPUKernel(const unsigned int* src, unsigned int* dest, const unsigned int* dest_above, int width_, int height_)
	{
		//CUDA_KERNEL_LOOP(index, height_)
		for (int index = 0; index < height_; index++)
		{
			if (index == 0)
			{
				for (int i = 1; i < width_; i++)
				{
					dest[i] = src[i] + dest[i - 1];
				}
			}
			else
			{
				int offset = index * width_;
				int s = 0;
				for (int i = 0; i < width_; i++)
				{
					s += src[offset + i];
					dest[offset + i] = s + dest_above[offset - width_ + i];
				}
			}
		}
	}

	void LABFeatureMap::IntegralGPU(std::shared_ptr<ImageTensor<int>> data)
	{
		const int* src = data->gpu_data();
		int* dest = data->mutable_gpu_data();
		const int* dest_above = dest;
		IntegralGPUKernel << </*CUDA_GET_BLOCKS(height_), CUDA_NUM_THREADS */1, 1>> >
			(src, dest, dest_above, width_, height_);
	}

	void LABFeatureMap::IntegralGPU(std::shared_ptr<ImageTensor<unsigned int>> data)
	{
		const unsigned int* src = data->gpu_data();
		unsigned int* dest = data->mutable_gpu_data();
		const unsigned int* dest_above = dest;
		IntegralGPUKernel << </*CUDA_GET_BLOCKS(height_), CUDA_NUM_THREADS*/1, 1 >> >
			(src, dest, dest_above, width_, height_);
	}

	void LABFeatureMap::ComputeRectSumGPU()
	{
		int width = width_ - rect_width_;
		int height = height_ - rect_height_;
		const int* int_img = int_img_->gpu_data();
		int* rect_sum = rect_sum_->mutable_gpu_data();
		//std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
		fix_rect_sum << <1, 1 >> > (rect_sum, int_img, rect_width_, rect_height_, width_);
		/*std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
		std::cout << "total detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl << std::endl;*/
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
		/*auto aaa = feat_map_->cpu_data();
		std::cout << "GPU code: " << std::endl;
		for (int i = 0; i < 100; i++)
		{
			std::cout << (int)aaa[i] << " ";
		}*/
	}
}

#endif