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
				for (int c = 0; c < src->channels(); c++)
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

	template <typename Dtype>
	void tensoroperation<Dtype>::flip_cpu(std::shared_ptr<tensor<Dtype>> src, 
		std::shared_ptr<tensor<Dtype>>& dst, std::string axis)
	{
		int channels = src->channels();
		int height = src->height();
		int width = src->width();
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device()));
		const Dtype* src_data = src->cpu_data();
		Dtype* dst_data = dst->mutable_cpu_data();
		if (axis=="x" || axis=="y")
		{
			for (int c = 0; c < channels; c++) 
			{
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						dst_data[((c * height + h) * width) + w] =
							src_data[((c * height + (axis == "y" ? (height - 1 - h) : h)) * width) + (axis == "x" ? (width - 1 - w) : w)];
					}
				}
			}
		}
		else if (axis=="c")
		{
			return;
		}
		else
		{
			return;
		}
	}

	template <typename Dtype>
	void tensoroperation<Dtype>::rgb2gray_cpu(std::shared_ptr<tensor<Dtype>> src, 
		std::shared_ptr<tensor<Dtype>>& dst)
	{
		if (src->channels() != 3)
		{
			return;
		}
		else
		{
			int height = src->height();
			int width = src->width();
			dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device()));
			int channel_offset = height*width;
			const Dtype* src_data = src->cpu_data();
			Dtype* dst_data = dst->mutable_cpu_data();
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					dst_data[i*width + j] = (src_data[channel_offset * 0 + i*width + j] + src_data[channel_offset * 1 + i*width + j] + src_data[channel_offset * 2 + i*width + j]) / 3.0f;
				}
			}
		}
	}

	template <typename Dtype>
	void tensoroperation<Dtype>::copy_cut_border_image_cpu(std::shared_ptr<tensor<Dtype>> src, 
		std::shared_ptr<tensor<Dtype>>& dst, int top, int left)
	{
		int w = dst->width();
		int h = dst->height();

		const Dtype* ptr = src->cpu_data() + src->width() * top + left;
		Dtype* outptr = dst->mutable_cpu_data();

		for (int y = 0; y < h; y++)
		{
			if (w < 12)
			{
				for (int x = 0; x < w; x++)
				{
					outptr[x] = ptr[x];
				}
			}
			else
			{
				memcpy(outptr, ptr, w * sizeof(Dtype));
			}
			outptr += w;
			ptr += src->width();
		}
	}

	template <typename Dtype>
	void tensoroperation<Dtype>::copy_cut_border_cpu(std::shared_ptr<tensor<Dtype>> src,
		std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right)
	{
		int w = src->width() - left - right;
		int h = src->height() - top - bottom;

		if (w == src->width() && h == src->height())
		{
			dst = src;
			return;
		}
		int channels = src->channels();

		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, h, w}, src->device()));
		if (dst->empty())
			return;

		// unroll image channel
#pragma omp parallel for
		for (int q = 0; q<channels; q++)
		{
			std::shared_ptr<tensor<Dtype>> cutm = std::make_shared<tensor<Dtype>>(dst->channel_tensor_ptr(q));
			copy_cut_border_image_cpu(std::make_shared<tensor<Dtype>>(src->channel_tensor_ptr(q)), cutm, top, left);
		}
	}

	template <typename Dtype>
	void tensoroperation<Dtype>::copy_make_border_image_cpu(std::shared_ptr<tensor<Dtype>> src, 
		std::shared_ptr<tensor<Dtype>>& dst, int top, int left, int type, Dtype v)
	{
		int w = dst->width();
		int h = dst->height();
		const Dtype* ptr = src->cpu_data();
		Dtype* outptr = dst->mutable_cpu_data();
		if (type == BORDER_CONSTANT)
		{
			int y = 0;
			// fill top
			for (; y < top; y++)
			{
				int x = 0;
				for (; x < w; x++)
				{
					outptr[x] = v;
				}
				outptr += w;
			}
			// fill center
			for (; y < (top + src->height()); y++)
			{
				int x = 0;
				for (; x < left; x++)
				{
					outptr[x] = v;
				}
				if (src->width() < 12)
				{
					for (; x < (left + src->width()); x++)
					{
						outptr[x] = ptr[x - left];
					}
				}
				else
				{
					memcpy(outptr + left, ptr, src->width() * sizeof(Dtype));
					x += src->width();
				}
				for (; x < w; x++)
				{
					outptr[x] = v;
				}
				ptr += src->width();
				outptr += w;
			}
			// fill bottom
			for (; y < h; y++)
			{
				int x = 0;
				for (; x < w; x++)
				{
					outptr[x] = v;
				}
				outptr += w;
			}
		}
		else if (type == BORDER_REPLICATE)
		{
			int y = 0;
			// fill top
			for (; y < top; y++)
			{
				int x = 0;
				for (; x < left; x++)
				{
					outptr[x] = ptr[0];
				}
				if (src->width() < 12)
				{
					for (; x < (left + src->width()); x++)
					{
						outptr[x] = ptr[x - left];
					}
				}
				else
				{
					memcpy(outptr + left, ptr, src->width() * sizeof(Dtype));
					x += src->width();
				}
				for (; x < w; x++)
				{
					outptr[x] = ptr[src->width() - 1];
				}
				outptr += w;
			}
			// fill center
			for (; y < (top + src->height()); y++)
			{
				int x = 0;
				for (; x < left; x++)
				{
					outptr[x] = ptr[0];
				}
				if (src->width() < 12)
				{
					for (; x < (left + src->width()); x++)
					{
						outptr[x] = ptr[x - left];
					}
				}
				else
				{
					memcpy(outptr + left, ptr, src->width() * sizeof(Dtype));
					x += src->width();
				}
				for (; x < w; x++)
				{
					outptr[x] = ptr[src->width() - 1];
				}
				ptr += src->width();
				outptr += w;
			}
			// fill bottom
			ptr -= src->width();
			for (; y < h; y++)
			{
				int x = 0;
				for (; x < left; x++)
				{
					outptr[x] = ptr[0];
				}
				if (src->width() < 12)
				{
					for (; x < (left + src->width()); x++)
					{
						outptr[x] = ptr[x - left];
					}
				}
				else
				{
					memcpy(outptr + left, ptr, src->width() * sizeof(Dtype));
					x += src->width();
				}
				for (; x < w; x++)
				{
					outptr[x] = ptr[src->width() - 1];
				}
				outptr += w;
			}
		}
	}

	template <typename Dtype>
	void tensoroperation<Dtype>::copy_make_border_cpu(std::shared_ptr<tensor<Dtype>> src, 
		std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right, int type, Dtype v)
	{
		int w = src->width() + left + right;
		int h = src->height() + top + bottom;
		if (w == src->width() && h == src->height())
		{
			dst = src;
			return;
		}
		int channels = src->channels();

		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, h, w}, src->device()));
		if (dst->empty())
			return;

		// unroll image channel
		//#pragma omp parallel for
		for (int q = 0; q<channels; q++)
		{
			std::shared_ptr<tensor<Dtype>> borderm = std::make_shared<tensor<Dtype>>(dst->channel_tensor_ptr(q));
			copy_make_border_image_cpu(std::make_shared<tensor<Dtype>>(src->channel_tensor_ptr(q)), borderm, top, left, type, v);
		}
	}


	template class tensoroperation<float>;
	template class tensoroperation<int>;
	template class tensoroperation<char>;
	template class tensoroperation<unsigned char>;
}
