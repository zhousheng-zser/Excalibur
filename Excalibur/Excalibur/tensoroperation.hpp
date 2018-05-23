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
	public:
		tensoroperation(){};
		~tensoroperation(){};

		enum interpolationType { Nearest, Bilinear, Cubic };
		enum borderType { BORDER_CONSTANT, BORDER_REPLICATE };

		template <typename Dtype>
		static void resize_cpu(std::shared_ptr<tensor<Dtype>> src,
			std::shared_ptr<tensor<Dtype>>& dst, int new_height, int new_width, int type)
		{
			CHECK_EQ(src->num(), 1);
			if (new_height*new_width<=0)
			{
				LOG(ERROR) << "Illegal input size.";
				return;
			}
			int old_height = src->height();
			int old_width = src->width();
			if (new_width==old_width&&new_height==old_height)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}
			int channels = src->channels();
			dst.reset(new tensor<Dtype>(std::vector<int>{src->num(), channels,
				new_height, new_width}, src->device()));
			Dtype* dst_data = dst->mutable_cpu_data();
			const Dtype* src_data = src->cpu_data();
			if (type == Nearest)
			{
				resize_cpu_nearset(src_data, old_height, old_width, channels, dst_data, new_height, new_width);
			}
			else if (type == Bilinear)
			{
				resize_cpu_bilinear(src_data, old_height, old_width, channels, dst_data, new_height, new_width);
			}
			else
			{
				LOG(ERROR) << "Un-support interpolation type.";
				return;
			}
		}

		template <typename Dtype>
		static void rotate_cpu(std::shared_ptr<tensor<Dtype>> src,
			std::shared_ptr<tensor<Dtype>>& dst, float theta, int center_x, int center_y, int type, Dtype v)
		{
			CHECK_EQ(src->num(), 1);
			int height = src->height();
			int width = src->width();
			if (fabs(theta)<0.000001)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}
			int channels = src->channels();
			dst.reset(new tensor<Dtype>(std::vector<int>{src->num(), channels,
				height, width}, src->device()));
			Dtype* dst_data = dst->mutable_cpu_data();
			const Dtype* src_data = src->cpu_data();
			if (type == Nearest)
			{
				rotate_cpu_nearset(src->cpu_data(), height, width, channels,
					dst->mutable_cpu_data(), theta, center_x, center_y, v);
			}
			else if (type == Bilinear)
			{
				
			}
			else
			{
				LOG(ERROR) << "Un-support interpolation type.";
				return;
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
			CHECK_EQ(src->num(), 1);
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
			CHECK_EQ(src->num(), 1);
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
				copy_make_border_image_cpu(src->cpu_data() + q * src->width() * src->height(),
					src->width(), src->height(), src->channels(), dst->mutable_cpu_data() + q*dst->width()*dst->height(),
					dst->height(), dst->width(), top, left, type, v);
			}
		}

		template <typename Dtype>
		static void copy_cut_border_cpu(std::shared_ptr<tensor<Dtype>> src, 
			std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right)
		{
			CHECK_EQ(src->num(), 1);
			int w = src->width() - left - right;
			int h = src->height() - top - bottom;

			if (w == src->width() && h == src->height())
			{
				dst = src;
				LOG(WARNING) << "Just copy from the source.";
				return;
			}
			if (w<=0||h<=0)
			{
				LOG(ERROR) << "Illegal input size.";
				return;
			}
			int channels = src->channels();

			dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, h, w}, src->device()));

			copy_cut_border_image_cpu(src->cpu_data(), src->height(), src->width(), src->channels(),
				dst->mutable_cpu_data(), dst->height(), dst->width(), top, left);
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
		static void copy_make_border_image_cpu(const Dtype* src_data, int src_height, int src_width, int channels,
			Dtype* dst_data, int dst_height, int dst_width, int top, int left, int type, Dtype v)
		{
			int w = dst_width;
			int h = dst_height;

			if (type == BORDER_CONSTANT)
			{
				int y = 0;
				// fill top
				for (; y < top; y++)
				{
					int x = 0;
					for (; x < w; x++)
					{
						dst_data[x] = v;
					}
					dst_data += w;
				}
				// fill center
				for (; y < (top + src_height); y++)
				{
					int x = 0;
					for (; x < left; x++)
					{
						dst_data[x] = v;
					}
					if (src_width < 12)
					{
						for (; x < (left + src_width); x++)
						{
							dst_data[x] = src_data[x - left];
						}
					}
					else
					{
						memcpy(dst_data + left, src_data, src_width * sizeof(Dtype));
						x += src_width;
					}
					for (; x < w; x++)
					{
						dst_data[x] = v;
					}
					src_data += src_width;
					dst_data += w;
				}
				// fill bottom
				for (; y < h; y++)
				{
					int x = 0;
					for (; x < w; x++)
					{
						dst_data[x] = v;
					}
					dst_data += w;
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
						dst_data[x] = src_data[0];
					}
					if (src_width < 12)
					{
						for (; x < (left + src_width); x++)
						{
							dst_data[x] = src_data[x - left];
						}
					}
					else
					{
						memcpy(dst_data + left, src_data, src_width * sizeof(Dtype));
						x += src_width;
					}
					for (; x < w; x++)
					{
						dst_data[x] = src_data[src_width - 1];
					}
					dst_data += w;
				}
				// fill center
				for (; y < (top + src_height); y++)
				{
					int x = 0;
					for (; x < left; x++)
					{
						dst_data[x] = src_data[0];
					}
					if (src_width < 12)
					{
						for (; x < (left + src_width); x++)
						{
							dst_data[x] = src_data[x - left];
						}
					}
					else
					{
						memcpy(dst_data + left, src_data, src_width * sizeof(Dtype));
						x += src_width;
					}
					for (; x < w; x++)
					{
						dst_data[x] = src_data[src_width - 1];
					}
					src_data += src_width;
					dst_data += w;
				}
				// fill bottom
				src_data -= src_width;
				for (; y < h; y++)
				{
					int x = 0;
					for (; x < left; x++)
					{
						dst_data[x] = src_data[0];
					}
					if (src_width < 12)
					{
						for (; x < (left + src_width); x++)
						{
							dst_data[x] = src_data[x - left];
						}
					}
					else
					{
						memcpy(dst_data + left, src_data, src_width * sizeof(Dtype));
						x += src_width;
					}
					for (; x < w; x++)
					{
						dst_data[x] = src_data[src_width - 1];
					}
					dst_data += w;
				}
			}
		}

		template <typename Dtype>
		static void copy_cut_border_image_cpu(const Dtype* src_data, int src_height, int src_width, int channels,
			Dtype* dst_data, int dst_height, int dst_width, int top, int left)
		{
			int dst_offset = dst_width * dst_height;
			// unroll image channel
			//#pragma omp parallel for
			for (int c = 0; c < channels; c++)
			{
				int c_dst_offset = c * dst_offset;
				const Dtype* ptr = src_data + c * src_width * src_height + src_width * top + left;
				for (int h = 0; h < dst_height; h++)
				{
					if (dst_width < 12)
					{
						int dst_sub_offset = h * dst_width;
						for (int x = 0; x < dst_width; x++)
						{
							dst_data[x + c_dst_offset + dst_sub_offset] = ptr[x];
						}
					}
					else
					{
						memcpy(dst_data + c_dst_offset + h * dst_width, ptr, dst_width * sizeof(Dtype));
					}
					ptr += src_width;
				}
			}
		}

		template <typename Dtype>
		static void resize_cpu_nearset(const Dtype* src_data, int old_height, int old_width, int channels,
			Dtype* dst_data, int new_height, int new_width)
		{
			float width_ratio = old_width * 1.0f / new_width;
			float height_ratio = old_height * 1.0f / new_height;
			int src_offset = old_height * old_width;
			int dst_offset = new_width * new_height;
			int* c_dst_offset = new int[channels];
			int* c_src_offset = new int[channels];
			for (int c = 0; c < channels; c++)
			{
				c_dst_offset[c] = c*dst_offset;
				c_src_offset[c] = c*src_offset;
			}
			for (int c = 0; c < channels; c++)
			{
				for (int h = 0; h < new_height; h++)
				{
					int y0 = int(h * height_ratio);
					int dst_sub_offset = h * new_width;
					int src_sub_offset = y0 * old_width;
					for (int w = 0; w < new_width; w++)
					{
						int x0 = int(w * width_ratio);
						dst_data[c_dst_offset[c] + dst_sub_offset + w] =
							src_data[c_src_offset[c] + src_sub_offset + x0];
					}
				}
			}
			delete c_dst_offset;
			delete c_src_offset;
		}

		template <typename Dtype>
		static void resize_cpu_bilinear(const Dtype* src_data, int old_height, int old_width, int channels,
			Dtype* dst_data, int new_height, int new_width)
		{
			float x_ratio = ((float)(old_width - 1)) / new_width;
			float y_ratio = ((float)(old_height - 1)) / new_height;
			int src_offset = old_height * old_width;
			int dst_offset = new_width * new_height;
			float x_diff, y_diff;
			int* c_src_offset = new int[channels];
			int* c_dst_offset = new int[channels];
			for (int c = 0; c < channels; c++)
			{
				c_src_offset[c] = c * src_offset;
				c_dst_offset[c] = c * dst_offset;
			}
			for (int h = 0; h < new_height; h++)
			{
				int dst_sub_offset = h * new_width;
				for (int w = 0; w < new_width; w++)
				{
					int x = (int)(x_ratio * w);
					int y = (int)(y_ratio * h);
					x_diff = x_ratio * w - x;
					y_diff = y_ratio * h - y;
					int index = y*old_width + x;
					for (int c = 0; c < channels; c++)
					{
						Dtype A = src_data[c_src_offset[c] + index];
						Dtype B = src_data[c_src_offset[c] + index + 1];
						Dtype C = src_data[c_src_offset[c] + index + old_width];
						Dtype D = src_data[c_src_offset[c] + index + old_width + 1];

						dst_data[c_dst_offset[c] + dst_sub_offset + w]
							= Dtype(static_cast<float>(A) *(1 - x_diff)*(1 - y_diff) +
								static_cast<float>(B)*x_diff*(1 - y_diff) +
								static_cast<float>(C)*y_diff*(1 - x_diff) +
								static_cast<float>(D)*x_diff*y_diff);
					}
				}
			}
			delete c_src_offset;
		}

		template <typename Dtype>
		static void rotate_cpu_nearset(const Dtype* src_data, int height, int width, int channels,
			Dtype* dst_data, float theta, int center_x, int center_y, Dtype v)
		{
			memset(dst_data, v, height * width * channels * sizeof(Dtype));
			float SinTheta = sin(theta);
			float CosTheta = cos(theta);
			float ConstX = -center_x*CosTheta + center_y*SinTheta + center_x + 0.5;
			float ConstY = -center_y*CosTheta - center_x*SinTheta + center_y + 0.5;
			int offset = height * width;
			for (int y = 0; y<height; y++)
			{
				float x1 = -y*SinTheta - CosTheta + ConstX;
				float y1 = y*CosTheta - SinTheta + ConstY;
				for (int x = 0; x<width; x++)
				{
					x1 += CosTheta;
					y1 += SinTheta;

					int x0 = int(x1);
					int y0 = int(y1);
					if (x0<0 || x0>width - 1 || y0<0 || y0>height - 1)
					{
						continue;
					}
					for (int c = 0; c < channels; c++)
					{
						dst_data[c * offset + y * width + x] = src_data[c * offset + y0 * width + x0];
					}
				}
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
