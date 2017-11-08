#include "graphicoperations.hpp"

#define get_index(x, y, offset) (y*offset+x) 

namespace flaskcv
{
	graphicoperations::graphicoperations()
	{
	}


	graphicoperations::~graphicoperations()
	{
	}

	void graphicoperations::bilinear_resize_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int new_height, int new_width)
	{
		int old_height = src->height();
		int old_width = src->width();
		dst.reset(new graphic(src->channel(), new_height, new_width));
		float* dst_data = dst->mutable_cpu_data();
		const float* src_data = src->cpu_data();
		int width_offset = new_width;
		int new_channel_offset = new_height*new_width;
		int old_channel_offset = old_height*old_width;
		for (int i = 0; i < new_height; i++)
		{
			float* p = dst_data + i * width_offset;
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
					p[j + c*new_channel_offset] = (int)(src_data[get_index(fx, fy, old_width) + c*new_channel_offset] * x1*y1 + src_data[get_index(fx + 1, fy, old_width) + c*new_channel_offset] * x2*y1 +
						src_data[get_index(fx, fy + 1, old_width) + c*new_channel_offset] * x1*y2 + src_data[get_index(fx + 1, fy + 1, old_width) + c*new_channel_offset] * x2*y2) >> 22;
				}
			}
		}
	}

	void graphicoperations::flip_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, bool flip_height, bool flip_width)
	{
		int channels = src->channel();
		int height = src->height();
		int width = src->width();
		dst.reset(new graphic(channels, height, width));
		const float* src_data = src->cpu_data();
		float* dst_data = dst->mutable_cpu_data();
		for (int c = 0; c < channels; c++) {
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					dst_data[((c * height + h) * width) + w] =
						src_data[((c * height + (flip_height ? (height - 1 - h) : h)) * width) + (flip_width ? (width - 1 - w) : w)];
				}
			}
		}
	}

	void graphicoperations::rgb2gray_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst)
	{
		if (src->channel()!=3)
		{
			return;
		}
		else
		{
			int height = src->height();
			int width = src->width();
			dst.reset(new graphic(1, height, width));
			int channel_offset = height*width;
			const float* src_data = src->cpu_data();
			float* dst_data = dst->mutable_cpu_data();
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					dst_data[i*width + j] = (src_data[channel_offset * 0 + i*width + j] + src_data[channel_offset * 1 + i*width + j] + src_data[channel_offset * 2 + i*width + j]) / 3;
				}
			}
		}
	}

	void graphicoperations::copy_make_border_image_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int top, int left, int type, float v)
	{
		int w = dst->width();
		int h = dst->height();
		const float* ptr = src->cpu_data();
		float* outptr = dst->mutable_cpu_data();
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
					memcpy(outptr + left, ptr, src->width() * sizeof(float));
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
					memcpy(outptr + left, ptr, src->width() * sizeof(float));
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
					memcpy(outptr + left, ptr, src->width() * sizeof(float));
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
					memcpy(outptr + left, ptr, src->width() * sizeof(float));
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

	void graphicoperations::copy_make_border_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int top, int bottom, int left, int right, int type, float v)
	{
		int w = src->width() + left + right;
		int h = src->height() + top + bottom;
		if (w == src->width() && h == src->height())
		{
			dst = src;
			return;
		}
		int channels = src->channel();

		dst.reset(new graphic(channels, h, w));
		if (dst->empty())
			return;

		// unroll image channel
//#pragma omp parallel for
		for (int q = 0; q<channels; q++)
		{
			std::shared_ptr<graphic> borderm = std::make_shared<graphic>(dst->channel_graphic_ptr(q));
			copy_make_border_image_cpu(std::make_shared<graphic>(src->channel_graphic_ptr(q)), borderm, top, left, type, v);
		}
	}

	void graphicoperations::copy_cut_border_image_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int top, int left)
	{
		int w = dst->width();
		int h = dst->height();

		const float* ptr = src->cpu_data() + src->width() * top + left;
		float* outptr = dst->mutable_cpu_data();

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
				memcpy(outptr, ptr, w * sizeof(float));
			}
			outptr += w;
			ptr += src->width();
		}
	}


	void graphicoperations::copy_cut_border_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int top, int bottom, int left, int right)
	{
		int w = src->width() - left - right;
		int h = src->height() - top - bottom;

		if (w == src->width() && h == src->height())
		{
			dst = src;
			return;
		}
		int channels = src->channel();

		dst.reset(new graphic(channels, h, w));
		if (dst->empty())
			return;

		// unroll image channel
#pragma omp parallel for
		for (int q = 0; q<channels; q++)
		{
			std::shared_ptr<graphic> cutm =std::make_shared<graphic>(dst->channel_graphic_ptr(q));
			copy_cut_border_image_cpu(std::make_shared<graphic>(src->channel_graphic_ptr(q)), cutm, top, left);
		}
	}

}
