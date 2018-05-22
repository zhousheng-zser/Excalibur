#ifndef _TENSOROPERATION_HPP_
#define _TENSOROPERATION_HPP_

#include "tensor.hpp"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace excalibur
{
	class tensoroperation
	{
#define Get_Index(x, y, offset) (y*offset+x) 
		enum bordertype { BORDER_CONSTANT, BORDER_REPLICATE };
	public:
		tensoroperation(){};
		~tensoroperation(){};

		//Bug!
		template <typename Dtype>
		static void bilinear_resize_cpu(std::shared_ptr<tensor<Dtype>> src, 
			std::shared_ptr<tensor<Dtype>>& dst, int new_height, int new_width)
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
		static void flip_cpu(std::shared_ptr<tensor<Dtype>> src, 
			std::shared_ptr<tensor<Dtype>>& dst, std::string axis)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device()));
			const Dtype* src_data = src->cpu_data();
			Dtype* dst_data = dst->mutable_cpu_data();
			if (axis == "x" || axis == "y")
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
			else if (axis == "c")
			{
				return;
			}
			else
			{
				return;
			}
		}

		template <typename Dtype>
		static void rgb2gray_cpu(std::shared_ptr<tensor<Dtype>> src, 
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
		static void copy_make_border_cpu(std::shared_ptr<tensor<Dtype>> src, 
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

		template <typename Dtype>
		static void copy_cut_border_cpu(std::shared_ptr<tensor<Dtype>> src, 
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

#ifdef USE_OPENCV
		template <typename Dtype>
		static void convert2mat(std::shared_ptr<tensor<Dtype>> src, cv::Mat& dst)
		{
			CHECK_EQ(src->num(), 1);
			int channel = src->channels();
			if (channel>4)
			{
				LOG(ERROR) << "Too many channels.";
				return;
			}
			int width = src->width();
			int height = src->height();
			int type = get_cv_type<Dtype>();
			if (type<0)
			{
				LOG(ERROR) << "Un-support data type.";
				return;
			}
			dst = cv::Mat(height, width, CV_MAKETYPE(type, channel));
			const Dtype* src_data = src->cpu_data();
			int src_offset = width * height;
			int* c_src_offset = new int[channel];
			for (int c = 0; c < channel; c++)
			{
				c_src_offset[c] = c * src_offset;
			}
			for (int h = 0; h < height; h++)
			{
				Dtype* dst_data = dst.ptr<Dtype>(h);
				int src_sub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					for (int c = 0; c < channel; c++)
					{
						dst_data[w*channel + c] = src_data[c_src_offset[c] + src_sub_offset + w];
					}
				}
			}
			delete c_src_offset;
		}

		template <typename Dtype>
		static void convert2tensor(const cv::Mat src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			int channel = src.channels();
			int width = src.cols;
			int height = src.rows;
			if (src.data == nullptr)
			{
				LOG(ERROR) << "No data.";
				return;
			}
			int type_id = src.type() % 8;
			auto type_name = std::string(typeid(Dtype).name());
			if (type_id == 0)
			{
				if (type_name != std::string("unsigned char"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 1)
			{
				if (type_name != std::string("char"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 4)
			{
				if (type_name != std::string("int"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 5)
			{
				if (type_name != std::string("float"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else
			{
				LOG(ERROR) << "Un-support data type.";
				return;
			}
			dst.reset(new tensor<Dtype>(std::vector<int>{1, channel, height, width}, -1));
			Dtype* dst_data = dst->mutable_cpu_data();
			int dst_offset = width * height;
			int* c_dst_offset = new int[channel];
			for (int c = 0; c < channel; c++)
			{
				c_dst_offset[c] = c * dst_offset;
			}
			for (int c = 0; c < channel; c++)
			{
				for (int h = 0; h < height; h++)
				{
					const Dtype* src_data = src.ptr<Dtype>(h);
					int dst_sub_offset = h * width;
					for (int w = 0; w < width; w++)
					{
						dst_data[c_dst_offset[c] + dst_sub_offset + w] =
							src_data[w * channel + c];
					}
				}
			}
			delete c_dst_offset;
		}
#endif

	private:
		template <typename Dtype>
		static void copy_make_border_image_cpu(std::shared_ptr<tensor<Dtype>> src,
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
		static void copy_cut_border_image_cpu(std::shared_ptr<tensor<Dtype>> src,
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

#ifdef USE_OPENCV
		template <typename Dtype>
		static int get_cv_type()
		{
			auto name = typeid(Dtype).name();
			if (std::string("unsigned char") == std::string(name))
			{
				return 0;//CV_8U   0
			}
			else if (std::string("char") == std::string(name))
			{
				return 1;//CV_8S   1
			}
			else if (std::string("int") == std::string(name))
			{
				return 4;//CV_32S  4
			}
			else if (std::string("float") == std::string(name))
			{
				return 5;//CV_32F  5
			}
			else
			{
				return -1;
			}
		}
#endif
	};
}
#endif // !_TENSOROPERATION_HPP_
