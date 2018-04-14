#include "tensoroperation.hpp"

namespace excalibur
{
	template <typename Dtype>
	tensoroperation<Dtype>::tensoroperation()
	{
	}

	template <typename Dtype>
	tensoroperation<Dtype>::~tensoroperation()
	{
	}

	template <typename Dtype>
	void tensoroperation<Dtype>::bilinear_resize_cpu(std::shared_ptr<tensor<Dtype>> src, 
		std::shared_ptr<tensor<Dtype>> & dst, 
		int new_height, int new_width)
	{
		int old_height = src->height();
		int old_width = src->width();
		dst.reset(new tensor<Dtype>(std::vector<int>{src->num(), src->channels(), 
			new_height, new_width}, src->device()));
		Dtype* dst_data = dst->mutable_cpu_data();
		const Dtype* src_data = src->cpu_data();
		int width_offset = new_width;
		int new_channel_offset = new_height*new_width;
		//int old_channel_offset = old_height*old_width;
		for (int i = 0; i < new_height; i++)
		{
			Dtype* p = dst_data + i * width_offset;
			float x = (i + 0.5)*old_height / new_height - 0.5;
			int fx = (int)x;
			x -= fx;
			short x1 = (1.f - x) * 2048;
			short x2 = 2048 - x1;
			for (int j = 0; j < new_width; j++)
			{
				float y = (j + 0.5)*old_width / new_width - 0.5;
				int fy = (int)y;
				y -= fy;
				short y1 = (1.f - y) * 2048;
				short y2 = 2048 - y1;
				for (int c = 0; c < src->channel(); c++)
				{
					p[j + c*new_channel_offset] = 
						static_cast<Dtype>(
							static_cast<int>(
								static_cast<float>(src_data[Get_Index(fx, fy, old_width) + c*new_channel_offset]) * x1*y1 +
								static_cast<float>(src_data[Get_Index(fx + 1, fy, old_width) + c*new_channel_offset]) * x2*y1 +
								static_cast<float>(src_data[Get_Index(fx, fy + 1, old_width) + c*new_channel_offset]) * x1*y2 +
								static_cast<float>(src_data[Get_Index(fx + 1, fy + 1, old_width) + c*new_channel_offset]) * x2*y2
								) 
							>> 22);
				}
			}
		}
	}

}
