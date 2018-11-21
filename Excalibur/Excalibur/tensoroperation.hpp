#ifndef _TENSOROPERATION_HPP_
#define _TENSOROPERATION_HPP_

#include "tensor.hpp"
#include "tensor_utils.hpp"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <glasssix\timer.hpp>
//#ifdef _OPENMP
//#include <omp.h>
//#endif
#include <omp.h>

#define PI 3.141592
namespace excalibur
{
	class tensoroperation
	{
	public:
		tensoroperation(){};
		~tensoroperation() {};

		template <typename Dtype>
		static void nchw2nhwc(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst)
		{
			CHECK_EQ(src->num(), 1);
			CHECK_EQ(src->type(), NCHW);
			int height = src->height();
			int width = src->width();
			int channels = src->channels();
			int offset = height * width;

			dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), NHWC));
			Dtype* dst_data = dst->mutable_cpu_data();
			const Dtype* src_data = src->cpu_data();

			for (int ch = 0; ch < channels; ++ch)
			{
				int channel_offset = ch * offset;
				for (int row = 0; row < height; ++row)
				{
					int row_offset = row * width;
					for (int col = 0; col < width; ++col)
					{
						dst_data[(row_offset + col) * channels + ch] = src_data[channel_offset + row_offset + col];
					}
				}
			}
		}

		template <typename Dtype>
		static void nhwc2nchw(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst)
		{
			CHECK_EQ(src->num(), 1);
			CHECK_EQ(src->type(), NHWC);
			int height = src->height();
			int width = src->width();
			int channels = src->channels();
			int offset = height * width;

			dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), NCHW));
			Dtype* dst_data = dst->mutable_cpu_data();
			const Dtype* src_data = src->cpu_data();

			for (int ch = 0; ch < channels; ++ch)
			{
				int channel_offset = ch * offset;
				for (int row = 0; row < height; ++row)
				{
					int row_offset = row * width;
					for (int col = 0; col < width; ++col)
					{
						dst_data[channel_offset + row_offset + col] = src_data[(row_offset + col) * channels + ch];
					}
				}
			}
		}

		template <typename Dtype>
		static void resize_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst,
			int dst_height, int dst_width, interpolationType type = Bilinear)
		{
			if (dst_height * dst_width <= 0)
			{
				LOG(ERROR) << "Illegal input size.";
				return;
			}

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			if (dst_width == width && dst_height == height)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			float width_ratio = (float)(width - 1) / (dst_width - 1);
			float height_ratio = (float)(height - 1) / (dst_height - 1);

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "type is NCHW.";
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				int src_offset = height * width;
				int dst_offset = dst_height * dst_width;

				for (int ch = 0; ch < channels; ++ch)
				{
					int src_channel_offset = ch * src_offset;
					int dst_channel_offset = ch * dst_offset;

#pragma omp parallel for
					for (int row = 0; row < dst_height; ++row)
					{
						float yf = row * height_ratio;
						int y = (int)yf;
						float ydiff = yf - y;

						int src_pos1 = src_channel_offset + y * width;
						int dst_pos1 = dst_channel_offset + row * dst_width;

						for (int col = 0; col < dst_width; ++col)
						{
							float xf = col * width_ratio;
							int x = (int)xf;
							float xdiff = xf - x;

							int src_pos2 = src_pos1 + x;
							int dst_pos2 = dst_pos1 + col;

							if (type == Nearest)
							{
								dst_data[dst_pos2] = src_data[src_pos2];
							}
							else if (type == Bilinear)
							{
								int src_pos3 = src_pos2 + width;
								Dtype A = src_data[src_pos2];
								Dtype B = src_data[src_pos2 + 1];
								Dtype C = src_data[src_pos3];
								Dtype D = src_data[src_pos3 + 1];
								dst_data[dst_pos2] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
									static_cast<float>(B) * xdiff * (1 - ydiff) +
									static_cast<float>(C) * ydiff * (1 - xdiff) +
									static_cast<float>(D) * xdiff * ydiff);
							}
							else
							{
								LOG(ERROR) << "Un-support interpolation type.";
							}
						}
					}
				}
			}
			else
			{
				LOG(WARNING) << "type is NHWC.";
				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

#pragma omp parallel for
				for (int row = 0; row < dst_height; ++row)
				{
					float yf = row * height_ratio;
					int y = (int)yf;
					float ydiff = yf - y;

					int src_pos1 = y * width * channels;
					int dst_pos1 = row * dst_width * channels;

					for (int col = 0; col < dst_width; ++col)
					{
						float xf = col * width_ratio;
						int x = (int)xf;
						float xdiff = xf - x;

						int src_pos2 = src_pos1 + x * channels;
						int dst_pos2 = dst_pos1 + col * channels;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_pos3 = src_pos2 + ch;
							int dst_pos3 = dst_pos2 + ch;

							if (type == Nearest)
							{
								dst_data[dst_pos3] = src_data[src_pos3];
							}
							else if (type == Bilinear)
							{
								int src_pos4 = src_pos3 + width * channels;
								Dtype A = src_data[src_pos3];
								Dtype B = src_data[src_pos3 + channels];
								Dtype C = src_data[src_pos4];
								Dtype D = src_data[src_pos4 + channels];
								dst_data[dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
									static_cast<float>(B) * xdiff * (1 - ydiff) +
									static_cast<float>(C) * ydiff * (1 - xdiff) +
									static_cast<float>(D) * xdiff * ydiff);
							}
							else
							{
								LOG(ERROR) << "Un-support interpolation type.";
							}

						}
					}
				}
			}
		}

		template <typename Dtype>
		static void rotate_with_center_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
			float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear)
		{
			if (fabs(theta) <= 1e-6)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			float rad = -1 * theta * (PI / 180);//逆时针方向为正
			float cosa = cos(rad);
			float sina = sin(rad);

			dst_width = (int)(width * abs(cosa) + height * abs(sina));
			dst_height = (int)(width * abs(sina) + height * abs(cosa));

			float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
			float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "type is NCHW.";
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				int src_offset = height * width;
				int dst_offset = dst_height * dst_width;
				for (int ch = 0; ch < channels; ++ch)
				{
					int src_channel_offset = ch * src_offset;
					int dst_channel_offset = ch * dst_offset;

#pragma omp parallel for
					for (int row = 0; row < dst_height; ++row)
					{
						int dst_pos1 = dst_channel_offset + row * dst_width;
						for (int col = 0; col < dst_width; ++col)
						{
							float xf = cosa * col + sina * row + VarX;
							float yf = -sina * col + cosa * row + VarY;
							int x = (int)(xf);
							int y = (int)(yf);
							float xdiff = xf - x;
							float ydiff = yf - y;
							int src_pos1 = src_channel_offset + y * width + x;
							int dst_pos2 = dst_pos1 + col;
							if (x >= width || x < 0 || y >= height || y < 0)
							{
								dst_data[dst_pos2] = (Dtype)fill;
							}
							else
							{
								if (type == Nearest)
								{
									dst_data[dst_pos2] = src_data[src_pos1];
								}
								else if (type == Bilinear)
								{
									int src_pos2 = src_pos1 + width;
									Dtype A = src_data[src_pos1];
									Dtype B = src_data[src_pos1 + 1];
									Dtype C = src_data[src_pos2];
									Dtype D = src_data[src_pos2 + 1];
									dst_data[dst_pos2] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
										static_cast<float>(B) * xdiff * (1 - ydiff) +
										static_cast<float>(C) * ydiff * (1 - xdiff) +
										static_cast<float>(D) * xdiff * ydiff);
								}
								else
								{
									LOG(ERROR) << "Un-support interpolation type.";
								}
							}
						}
					}
				}
			}
			else
			{
				LOG(WARNING) << "type is NHWC.";
				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

#pragma omp parallel for
				for (int row = 0; row < dst_height; ++row)
				{
					int dst_pos1 = row * dst_width * channels;

					for (int col = 0; col < dst_width; ++col)
					{
						float xf = cosa * col + sina * row + VarX;
						float yf = -sina * col + cosa * row + VarY;
						int x = (int)(xf);
						int y = (int)(yf);
						float xdiff = xf - x;
						float ydiff = yf - y;
						int src_pos1 = (y * width + x) * channels;
						int dst_pos2 = dst_pos1 + col * channels;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_pos2 = src_pos1 + ch;
							int dst_pos3 = dst_pos2 + ch;

							if (x >= width || x < 0 || y >= height || y < 0)
							{
								dst_data[dst_pos3] = (Dtype)fill;
							}
							else
							{
								if (type == Nearest)
								{
									dst_data[dst_pos3] = src_data[src_pos2];
								}
								else if (type == Bilinear)
								{
									int src_pos3 = src_pos2 + width * channels;
									Dtype A = src_data[src_pos2];
									Dtype B = src_data[src_pos2 + channels];
									Dtype C = src_data[src_pos3];
									Dtype D = src_data[src_pos3 + channels];
									dst_data[dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
										static_cast<float>(B) * xdiff * (1 - ydiff) +
										static_cast<float>(C) * ydiff * (1 - xdiff) +
										static_cast<float>(D) * xdiff * ydiff);
								}
								else
								{
									LOG(ERROR) << "Un-support interpolation type.";
								}
							}
						}
					}
				}
			}
		}

		template <typename Dtype, typename Ptype>
		static void rotate_with_points_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst,
			                           const point<Ptype> &center, float theta, float scale, int fill = 0, interpolationType type = Bilinear)
		{
			if (fabs(theta) <= 1e-6)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			double rad = theta*(PI / 180);
			double cosa = cos(rad);
			double sina = sin(rad);

			double a = scale * cosa;
			double b = scale * sina;
			double M_data[9];
			M_data[0] = a;
			M_data[1] = b;
			M_data[2] = (1 - a) * (double)center.x - b * (double)center.y;
			M_data[3] = -1 * b;
			M_data[4] = a;
			M_data[5] = b * (double)center.x + (1 - a) * (double)center.y;
			M_data[6] = 0;
			M_data[7] = 0;
			M_data[8] = 1;

			cv::Mat M(3, 3, CV_64F, M_data);
			cv::Mat reverse_M(3, 3, CV_64F);
			cv::invert(M, reverse_M);
			double reverse_M_data[9];
			memcpy(reverse_M_data, reverse_M.data, 9 * sizeof(double));

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "type is NCHW.";
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				for (int ch = 0; ch < channels; ++ch)
				{
					int channel_offset = ch * offset;

#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						double temp_xf = reverse_M_data[1] * row + reverse_M_data[2];
						double temp_yf = reverse_M_data[4] * row + reverse_M_data[5];
						int temp_dst_index = channel_offset + row * width;

						for (int col = 0; col < width; ++col)
						{
							double xf = reverse_M_data[0] * col + temp_xf;
							double yf = reverse_M_data[3] * col + temp_yf;
							int x = (int)xf;
							int y = (int)yf;
							float xdiff = xf - x;
							float ydiff = yf - y;

							int src_index = channel_offset + y * width + x;
							int dst_index = temp_dst_index + col;

							if (x < 0 || x >= width || y < 0 || y >= height)
							{
								dst_data[dst_index] = (Dtype)fill;
							}
							else
							{
								if (type == Nearest)
								{
									dst_data[dst_index] = src_data[src_index];
								}
								else if (type == Bilinear)
								{
									Dtype A = src_data[src_index];
									Dtype B = src_data[src_index + 1];
									Dtype C = src_data[src_index + width];
									Dtype D = src_data[src_index + width + 1];
									dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
										static_cast<float>(B) * xdiff * (1 - ydiff) +
										static_cast<float>(C) * ydiff * (1 - xdiff) +
										static_cast<float>(D) * xdiff * ydiff);
								}
								else
								{
									LOG(ERROR) << "Un-support interpolation type.";
								}
							}
						}
					}
				}
			}
			else
			{
				LOG(WARNING) << "type is NHWC.";
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

#pragma omp parallel for
				for (int row = 0; row < height; ++row)
				{
					double temp_xf = reverse_M_data[1] * row + reverse_M_data[2];
					double temp_yf = reverse_M_data[4] * row + reverse_M_data[5];
					int dst_pos1 = row * width * channels;

					for (int col = 0; col < width; ++col)
					{
						double xf = reverse_M_data[0] * col + temp_xf;
						double yf = reverse_M_data[3] * col + temp_yf;
						int x = (int)xf;
						int y = (int)yf;
						float xdiff = xf - x;
						float ydiff = yf - y;

						int src_pos1 = (y * width + x) * channels;
						int dst_pos2 = dst_pos1 + col * channels;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_pos2 = src_pos1 + ch;
							int dst_pos3 = dst_pos2 + ch;

							if (x < 0 || x >= width || y < 0 || y >= height)
							{
								dst_data[dst_pos3] = (Dtype)fill;
							}
							else
							{
								if (type == Nearest)
								{
									dst_data[dst_pos3] = src_data[src_pos2];
								}
								else if (type == Bilinear)
								{
									int src_pos3 = src_pos2 + width * channels;

									Dtype A = src_data[src_pos2];
									Dtype B = src_data[src_pos2 + channels];
									Dtype C = src_data[src_pos3];
									Dtype D = src_data[src_pos3 + channels];
									dst_data[dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
										static_cast<float>(B) * xdiff * (1 - ydiff) +
										static_cast<float>(C) * ydiff * (1 - xdiff) +
										static_cast<float>(D) * xdiff * ydiff);
								}
								else
								{
									LOG(ERROR) << "Un-support interpolation type.";
								}
							}
						}
					}
				}
			}
		}

		template <typename Dtype, typename Rtype>
		static void draw_rectangle_cpu(std::shared_ptr<tensor<Dtype>>& dst, rectangle<Rtype> rect, color color_, int thickness = 2 )
		{
			CHECK_EQ(dst->num(), 1);
			int channels = dst->channels();
			int height = dst->height();
			int width = dst->width();
			int src_offset = height * width;
			Dtype* dst_data = dst->mutable_cpu_data();

			if (thickness <= 0)
			{
				LOG(WARNING) << "Zero or minus thickness, return without any changes.";
				return;
			}

			if (rect.w <= 0 || rect.h <= 0)
			{
				LOG(WARNING) << "rect.width and rect.height must be positive, return without any changes.";
				return;
			}

			if (rect.x < 0 || rect.x >= width || rect.y < 0 || rect.y >= height)
			{
				LOG(WARNING) << "top-left point out of image, return without any changes.";
				return;
			}

			if (rect.x + rect.w >= width || rect.y + rect.h >= height)
			{
				LOG(WARNING) << "right-bottom point out of image, return without any changes.";
				return;
			}

			if (channels == 1)
			{
				Dtype fill_color = Dtype(color_.g / 3 + color_.b / 3 + color_.r / 3);

				//top
				for (int row = rect.y; row < rect.y + thickness; ++row) 
				{
					Dtype* row_data = dst_data + row * width;
					memset(row_data + rect.x, fill_color, rect.w * sizeof(Dtype));
				}

				//bottom
				for (int row = rect.y + rect.h - thickness; row < rect.y + rect.h; ++row)
				{
					Dtype* row_data = dst_data + row * width;
					memset(row_data + rect.x, fill_color, rect.w * sizeof(Dtype));
				}

				//2-sides
				for (int row = rect.y; row < rect.y + rect.h; ++row) 
				{
					Dtype* row_data = dst_data + row * width;

					//left-side
					memset(row_data + rect.x, fill_color, thickness * sizeof(Dtype));

					//right-side
					memset(row_data + rect.x + rect.w - thickness, fill_color, thickness * sizeof(Dtype));
				}
			}
			else if (channels == 3)
			{
				Dtype* fill_color = new Dtype[3];
				fill_color[0] = color_.r;
				fill_color[1] = color_.g;
				fill_color[2] = color_.b;

				if (dst->type() == NCHW)
				{
					LOG(WARNING) << "type is NCHW.";
					for (int ch = 0; ch < 3; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						//top
						for (int row = rect.y; row < rect.y + thickness; ++row)
						{
							Dtype* row_data = dst_data + src_channel_offset + row * width;
							memset(row_data + rect.x, fill_color[ch], rect.w * sizeof(Dtype));
						}

						//bottom
						for (int row = rect.y + rect.h - thickness; row < rect.y + rect.h; ++row)
						{
							Dtype* row_data = dst_data + src_channel_offset + row * width;
							memset(row_data + rect.x, fill_color[ch], rect.w * sizeof(Dtype));
						}

						//2-sides
						for (int row = rect.y; row < rect.y + rect.h; ++row)
						{
							Dtype* row_data = dst_data + src_channel_offset + row * width;

							//left-side
							memset(row_data + rect.x, fill_color[ch], thickness * sizeof(Dtype));

							//right-side
							memset(row_data + rect.x + rect.w - thickness, fill_color[ch], thickness * sizeof(Dtype));
						}
					}
				}
				else
				{
					LOG(WARNING) << "type is NHWC.";

					//top
					for (int row = rect.y; row < rect.y + thickness; ++row)
					{
						int pos1 = (row * width + rect.x) * channels;
						
						for (int col = 0; col < rect.w; ++col)
						{
							int pos2 = pos1 + col * channels;
							for (int ch = 0; ch < 3; ++ch)
							{
								dst_data[pos2 + ch] = fill_color[ch];
							}
						}
					}

					//bottom
					for (int row = rect.y + rect.h - thickness; row < rect.y + rect.h; ++row)
					{
						int pos1 = (row * width + rect.x) * channels;
						
						for (int col = 0; col < rect.w; ++col)
						{
							int pos2 = pos1 + col * channels;
							for (int ch = 0; ch < 3; ++ch)
							{
								dst_data[pos2 + ch] = fill_color[ch];
							}
						}
					}

					//2-sides
					for (int row = rect.y; row < rect.y + rect.h; ++row)
					{
						int left_pos1 = (row * width + rect.x) * channels;
						int right_pos1 = (row * width + rect.x + rect.w - thickness) * channels;

						for (int col = 0; col < thickness; ++col)
						{
							int left_pos2 = left_pos1 + col * channels;
							int right_pos2 = right_pos1 + col * channels;

							for (int ch = 0; ch < 3; ++ch)
							{
								dst_data[left_pos2 + ch] = fill_color[ch];
								dst_data[right_pos2 + ch] = fill_color[ch];
							}
						}
					}
				}
				delete fill_color;
			}
			else
			{
				LOG(WARNING) << "Illegal channel numbers. Return without any changes.";
			}
		}

		template <typename Dtype, typename Rtype>
		static void draw_circle_cpu(std::shared_ptr<tensor<Dtype>>& dst, point<Rtype> center, int radius, color color_, int thickness = 2)
		{
			CHECK_EQ(dst->num(), 1);
			int channels = dst->channels();
			int height = dst->height();
			int width = dst->width();
			int offset = height * width;
			Dtype* dst_data = dst->mutable_cpu_data();

			if (center.x < radius || center.x >= width - radius && center.y < radius && center.y >= height - radius) 
			{
				LOG(WARNING) << "circle out of image, return without any changes.";
				return;
			}

			if (thickness > 0) 
			{
				if (dst->type() == NCHW)
				{
					LOG(WARNING) << "type is NCHW.";
				}
				else
				{
					LOG(WARNING) << "type is NHWC.";
				}

				for (int r = radius; r > radius - thickness; --r)
				{
					int dx = r, dy = 0, err = 0, plus = 1, minus = (r << 1) - 1;
					while (dx >= dy)
					{
						int mask;
						int y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
						int x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;

						if (channels == 1)
						{
							Dtype fill_color = Dtype(color_.g / 3 + color_.b / 3 + color_.r / 3);

							Dtype *tptr0 = dst_data + y11 * width;
							Dtype *tptr1 = dst_data + y12 * width;
							memset(tptr0 + x11, fill_color, 2 * sizeof(Dtype));
							memset(tptr1 + x11, fill_color, 2 * sizeof(Dtype));
							memset(tptr0 + x12, fill_color, 2 * sizeof(Dtype));
							memset(tptr1 + x12, fill_color, 2 * sizeof(Dtype));

							tptr0 = dst_data + y21 * width;
							tptr1 = dst_data + y22 * width;
							memset(tptr0 + x21, fill_color, 2 * sizeof(Dtype));
							memset(tptr1 + x21, fill_color, 2 * sizeof(Dtype));
							memset(tptr0 + x22, fill_color, 2 * sizeof(Dtype));
							memset(tptr1 + x22, fill_color, 2 * sizeof(Dtype));
						}
						else if (channels == 3)
						{
							Dtype* fill_color = new Dtype[3];
							fill_color[0] = color_.r;
							fill_color[1] = color_.g;
							fill_color[2] = color_.b;

							if (dst->type() == NCHW)
							{
								for (int ch = 0; ch < 3; ++ch)
								{
									int channel_offset = ch * offset;

									Dtype *tptr0 = dst_data + channel_offset + y11 * width;
									Dtype *tptr1 = dst_data + channel_offset + y12 * width;
									memset(tptr0 + x11, fill_color[ch], 2 * sizeof(Dtype));
									memset(tptr1 + x11, fill_color[ch], 2 * sizeof(Dtype));
									memset(tptr0 + x12, fill_color[ch], 2 * sizeof(Dtype));
									memset(tptr1 + x12, fill_color[ch], 2 * sizeof(Dtype));

									tptr0 = dst_data + channel_offset + y21 * width;
									tptr1 = dst_data + channel_offset + y22 * width;
									memset(tptr0 + x21, fill_color[ch], 2 * sizeof(Dtype));
									memset(tptr1 + x21, fill_color[ch], 2 * sizeof(Dtype));
									memset(tptr0 + x22, fill_color[ch], 2 * sizeof(Dtype));
									memset(tptr1 + x22, fill_color[ch], 2 * sizeof(Dtype));
								}
							}
							else
							{
								int pos_x11_y11 = (y11 * width + x11) * 3;
								int pos_x11_y12 = (y12 * width + x11) * 3;
								int pos_x12_y11 = (y11 * width + x12) * 3;
								int pos_x12_y12 = (y12 * width + x12) * 3;

								int pos_x21_y21 = (y21 * width + x21) * 3;
								int pos_x21_y22 = (y22 * width + x21) * 3;
								int pos_x22_y21 = (y21 * width + x22) * 3;
								int pos_x22_y22 = (y22 * width + x22) * 3;

								for (int col = 0; col < 2; ++col)//填充相邻2个像素
								{
									for (int ch = 0; ch < 3; ++ch)
									{
										dst_data[pos_x11_y11 + col * 3 + ch] = fill_color[ch];
										dst_data[pos_x11_y12 + col * 3 + ch] = fill_color[ch];
										dst_data[pos_x12_y11 + col * 3 + ch] = fill_color[ch];
										dst_data[pos_x12_y12 + col * 3 + ch] = fill_color[ch];

										dst_data[pos_x21_y21 + col * 3 + ch] = fill_color[ch];
										dst_data[pos_x21_y22 + col * 3 + ch] = fill_color[ch];
										dst_data[pos_x22_y21 + col * 3 + ch] = fill_color[ch];
										dst_data[pos_x22_y22 + col * 3 + ch] = fill_color[ch];
									}
								}
							}

							delete fill_color;
						}
						else
						{
							LOG(WARNING) << "Illegal channel numbers. Return without any changes.";
						}

						dy++;
						err += plus;
						plus += 2;

						mask = (err <= 0) - 1;

						err -= minus & mask;
						dx += mask;
						minus -= mask & 2;
					}
				}
			}
			else 
			{
				int dx = radius, dy = 0, err = 0, plus = 1, minus = (radius << 1) - 1;
				while (dx >= dy)
				{
					int mask;
					int y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
					int x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;

					if (channels == 1)
					{
						Dtype fill_color = Dtype(color_.g / 3 + color_.b / 3 + color_.r / 3);

						Dtype *tptr0 = dst_data + y11 * width;
						Dtype *tptr1 = dst_data + y12 * width;
						memset(tptr0 + x11, fill_color, (x12 - x11) * sizeof(Dtype));
						memset(tptr1 + x11, fill_color, (x12 - x11) * sizeof(Dtype));

						tptr0 = dst_data + y21 * width;
						tptr1 = dst_data + y22 * width;
						memset(tptr0 + x21, fill_color, (x22 - x21) * sizeof(Dtype));
						memset(tptr1 + x21, fill_color, (x22 - x21) * sizeof(Dtype));
					}
					else if (channels == 3)
					{
						Dtype* fill_color = new Dtype[3];
						fill_color[0] = color_.r;
						fill_color[1] = color_.g;
						fill_color[2] = color_.b;

						if (dst->type() == NCHW)
						{
							for (int ch = 0; ch < 3; ++ch)
							{
								int channel_offset = ch * offset;

								Dtype *tptr0 = dst_data + channel_offset + y11 * width;
								Dtype *tptr1 = dst_data + channel_offset + y12 * width;
								memset(tptr0 + x11, fill_color[ch], (x12 - x11) * sizeof(Dtype));
								memset(tptr1 + x11, fill_color[ch], (x12 - x11) * sizeof(Dtype));

								tptr0 = dst_data + channel_offset + y21 * width;
								tptr1 = dst_data + channel_offset + y22 * width;
								memset(tptr0 + x21, fill_color[ch], (x22 - x21) * sizeof(Dtype));
								memset(tptr1 + x21, fill_color[ch], (x22 - x21) * sizeof(Dtype));
							}
						}
						else
						{
							int pos_x11_y11 = (y11 * width + x11) * 3;
							int pos_x11_y12 = (y12 * width + x11) * 3;
							for (int col = 0; col < x12 - x11; ++col)
							{
								for (int ch = 0; ch < 3; ++ch)
								{
									dst_data[pos_x11_y11 + col * 3 + ch] = fill_color[ch];
									dst_data[pos_x11_y12 + col * 3 + ch] = fill_color[ch];
								}
							}

							int pos_x21_y21 = (y21 * width + x21) * 3;
							int pos_x21_y22 = (y22 * width + x21) * 3;
							for (int col = 0; col < x22 - x21; ++col)
							{
								for (int ch = 0; ch < 3; ++ch)
								{
									dst_data[pos_x21_y21 + col * 3 + ch] = fill_color[ch];
									dst_data[pos_x21_y22 + col * 3 + ch] = fill_color[ch];
								}
							}
						}

						delete fill_color;
					}
					else
					{
						LOG(WARNING) << "Illegal channel numbers. Return without any changes.";
					}

					dy++;
					err += plus;
					plus += 2;

					mask = (err <= 0) - 1;

					err -= minus & mask;
					dx += mask;
					minus -= mask & 2;
				}
			}
		}

		template <typename Dtype>
		static void flip_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, flipType axis = Width_Wise)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "type is NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				if (axis == Width_Wise)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

						for (int row = 0; row < height; ++row)
						{
							int index = channel_offset + row * width;
							for (int col = 0; col < width; ++col)
							{
								dst_data[index + col] = src_data[index + (width - col - 1)];
							}
						}
					}
				}
				else if (axis == Height_Wise)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

						for (int row = 0; row < height; ++row)
						{
							int dst_index = channel_offset + row * width;
							int src_index = channel_offset + (height - row - 1) * width;
							memcpy(dst_data + dst_index, src_data + src_index, width * sizeof(Dtype));
						}
					}
				}
				else if (axis == Center_Wise)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

						for (int row = 0; row < height; ++row)
						{
							int dst_index = channel_offset + row * width;
							int src_index = channel_offset + (height - row - 1) * width;
							for (int col = 0; col < width; ++col)
							{
								dst_data[dst_index + col] = src_data[src_index + (width - col - 1)];
							}
						}
					}
				}
				else if (axis == Channel_Wise)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int dst_channel_offset = ch * offset;
						int src_channel_offset = (channels - ch - 1) * offset;

						for (int row = 0; row < height; ++row)
						{
							int row_offset = row * width;
							int dst_index = dst_channel_offset + row_offset;
							int src_index = src_channel_offset + row_offset;
							memcpy(dst_data + dst_index, src_data + src_index, width * sizeof(Dtype));
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support flip type.";
				}
			}
			else
			{
				LOG(WARNING) << "type is NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				if (axis == Width_Wise)
				{
					for (int row = 0; row < height; ++row)
					{
						int row_pos = row * width * channels;
						for (int col = 0; col < width; ++col)
						{
							int dst_pos = row_pos + col * channels;
							int src_pos = row_pos + (width - col - 1) * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_pos + ch] = src_data[src_pos + ch];
							}
						}
					}
				}
				else if (axis == Height_Wise)
				{
					for (int row = 0; row < height; ++row)
					{
						int dst_pos1 = row * width * channels;
						int src_pos1 = (height - row - 1) * width * channels;
						for (int col = 0; col < width; ++col)
						{
							int dst_pos2 = dst_pos1 + col * channels;
							int src_pos2 = src_pos1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_pos2 + ch] = src_data[src_pos2 + ch];
							}
						}
					}
				}
				else if (axis == Center_Wise)
				{
					for (int row = 0; row < height; ++row)
					{
						int dst_pos1 = row * width * channels;
						int src_pos1 = (height - row - 1) * width * channels;
						for (int col = 0; col < width; ++col)
						{
							int dst_pos2 = dst_pos1 + col * channels;
							int src_pos2 = src_pos1 + (width - col - 1) * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_pos2 + ch] = src_data[src_pos2 + ch];
							}
						}
					}
				}
				else if (axis == Channel_Wise)
				{
					for (int row = 0; row < height; ++row)
					{
						int row_pos = row * width * channels;
						for (int col = 0; col < width; ++col)
						{
							int col_pos = row_pos + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[col_pos + ch] = src_data[col_pos + (channels - ch - 1)];
							}
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support flip type.";
				}
			}
		}

		template <typename Dtype>
		static void rgb2gray_cpu(const std::shared_ptr<tensor<Dtype>> &src,	std::shared_ptr<tensor<Dtype>>& dst)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			if (channels != 3)
			{
				LOG(ERROR) << "Incorrect input channel.";
				return;
			}

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "type is NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				for (int row = 0; row < height; ++row)
				{
					int index = row * width;
					for (int col = 0; col < width; ++col)
					{
						int pos = index + col;
						//opencv读取RGB图像后，以B、G、R的顺序进行存储
						//转换公式为:gray=0.114*B+0.587*G+0.299*R
						dst_data[pos] = Dtype(src_data[pos] * 0.114f +
							src_data[offset * 1 + pos] * 0.587f +
							src_data[offset * 2 + pos] * 0.299f);
					}
				}
			}
			else
			{
				LOG(WARNING) << "type is NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				for (int row = 0; row < height; ++row)
				{
					int dst_pos1 = row * width;
					int src_pos1 = row * width * channels;
					for (int col = 0; col < width; ++col)
					{
						int dst_pos2 = dst_pos1 + col;
						int src_pos2 = src_pos1 + col * channels;
						//opencv读取RGB图像后，以B、G、R的顺序进行存储
						//转换公式为:gray=0.114*B+0.587*G+0.299*R
						dst_data[dst_pos2] = Dtype(src_data[src_pos2] * 0.114f +
							src_data[src_pos2 + 1] * 0.587f +
							src_data[src_pos2 + 2] * 0.299f);
					}
				}
			}
		}

		template <typename Dtype>
		static void matrix_transpose_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "type is NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, width, height}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				for (int ch = 0; ch < channels; ++ch) {
					int channel_offset = ch * offset;
					for (int row = 0; row < width; ++row)
					{
						int dst_index = channel_offset + row * height;
						for (int col = 0; col < height; ++col)
						{
							dst_data[dst_index + col] = src_data[channel_offset + col * width + row];
						}
					}
				}
			}
			else 
			{
				LOG(WARNING) << "type is NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, width, height, channels}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				for (int row = 0; row < width; ++row)
				{
					int dst_pos1 = row * height * channels;;
					for (int col = 0; col < height; ++col)
					{
						int dst_pos2 = dst_pos1 + col * channels;
						int src_pos = (col * width + row) * channels;

						for (int ch = 0; ch < channels; ++ch) 
						{
							dst_data[dst_pos2 + ch] = src_data[src_pos + ch];
						}
					}
				}
			}
		}

		template <typename Dtype, typename Rtype>
		static void roi_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, rectangle<Rtype> rect)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;

			if (rect.x < 0 || rect.x + rect.w >= width || rect.y < 0 || rect.y + rect.h >= height) {
				LOG(WARNING) << "rect is out of image";
				return;
			}

			int dst_height = (int)rect.h;
			int dst_width = (int)rect.w;
			int dst_offset = dst_height * dst_width;

			if (dst_height == height && dst_width == width)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				for (int ch = 0; ch < channels; ++ch)
				{
					int src_channel_offset = ch * src_offset;
					int dst_channel_offset = ch * dst_offset;

					for (int row = 0; row < dst_height; ++row)
					{
						int src_index = src_channel_offset + (row + rect.y) * width + rect.x;
						int dst_index = dst_channel_offset + row * dst_width;

						memcpy(dst_data + dst_index, src_data + src_index, dst_width * sizeof(Dtype));
					}
				}
			}
			else
			{
				LOG(WARNING) << "NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				for (int row = 0; row < dst_height; ++row)
				{
					int dst_pos1 = row * dst_width * channels;
					int src_pos1 = (row + rect.y) * width * channels;

					for (int col = 0; col < dst_width; ++col)
					{
						int dst_pos2 = dst_pos1 + col * channels;
						int src_pos2 = src_pos1 + (rect.x + col) * channels;

						for (int ch = 0; ch < channels; ++ch)
						{
							dst_data[dst_pos2 + ch] = src_data[src_pos2 + ch];
						}
					}
				}
			}
		}

		template <typename Dtype>
		static void make_border_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
			int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;

			int dst_height = height + top + bottom;
			int dst_width = width + left + right;
			int dst_offset = dst_height * dst_width;

			if (top < 0 || bottom < 0 || left < 0 || right < 0) 
			{
				LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
				return;
			}

			if (dst_height == height && dst_width == width)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				if (type == Border_Constant)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						int dst_channel_offset = ch * dst_offset;

						//top
						memset(dst_data + dst_channel_offset, (Dtype)fill, top * dst_width * sizeof(Dtype));

						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index = src_channel_offset + (row - top) * width;
							int dst_index = dst_channel_offset + row * dst_width;

							memset(dst_data + dst_index, (Dtype)fill, left * sizeof(Dtype));
							memcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype));
							memset(dst_data + dst_index + left + width, (Dtype)fill, right * sizeof(Dtype));
						}

						//bottom
						memset(dst_data + dst_channel_offset + (top + height) * dst_width, (Dtype)fill, bottom * dst_width * sizeof(Dtype));
					}
				}
				else if (type == Border_Replicate)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						int dst_channel_offset = ch * dst_offset;

						//top
						for (int row = 0; row < top; ++row)
						{
							int dst_index = dst_channel_offset + row * dst_width;
							memset(dst_data + dst_index, src_data[src_channel_offset], left * sizeof(Dtype));
							memcpy(dst_data + dst_index + left, src_data + src_channel_offset, width * sizeof(Dtype));
							memset(dst_data + dst_index + left + width, src_data[src_channel_offset + width - 1], right * sizeof(Dtype));
						}

						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index = src_channel_offset + (row - top) * width;
							int dst_index = dst_channel_offset + row * dst_width;

							memset(dst_data + dst_index, src_data[src_index], left * sizeof(Dtype));
							memcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype));
							memset(dst_data + dst_index + left + width, src_data[src_index + width - 1], right * sizeof(Dtype));
						}

						//bottom
						for (int row = top + height; row < dst_height; ++row)
						{
							int src_index = src_channel_offset + (height - 1) * width;
							int dst_index = dst_channel_offset + row * dst_width;

							memset(dst_data + dst_index, src_data[src_index], left * sizeof(Dtype));
							memcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype));
							memset(dst_data + dst_index + left + width, src_data[src_index + width - 1], right * sizeof(Dtype));
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support border type.";
					return;
				}
			}
			else
			{
				LOG(WARNING) << "NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				if (type == Border_Constant)
				{
					//top
					memset(dst_data, (Dtype)fill, top * dst_width * channels * sizeof(Dtype));

					//center
					for (int row = top; row < top + height; ++row)
					{
						int src_pos = (row - top) * width * channels;

						//left
						int dst_pos1 = row * dst_width * channels;
						memset(dst_data + dst_pos1, (Dtype)fill, left * channels * sizeof(Dtype));

						//center
						int dst_pos2 = dst_pos1 + left * channels;
						memcpy(dst_data + dst_pos2, src_data + src_pos, width * channels * sizeof(Dtype));

						//right
						int dst_pos3 = dst_pos2 + width * channels;
						memset(dst_data + dst_pos3, (Dtype)fill, right * channels * sizeof(Dtype));
					}

					//bottom
					memset(dst_data + (top + height) * dst_width * channels, (Dtype)fill, bottom * dst_width * channels * sizeof(Dtype));
				}
				else if (type == Border_Replicate)
				{
					//top
					for (int row = 0; row < top; ++row)
					{
						int dst_pos1 = row * dst_width * channels;

						//left
						for (int col = 0; col < left; ++col)
						{
							int dst_index = dst_pos1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index + ch] = src_data[ch];
							}
						}
												
						//center
						int dst_pos2 = dst_pos1 + left * channels;
						memcpy(dst_data + dst_pos2, src_data, width * channels * sizeof(Dtype));
												
						//right
						int dst_pos3 = dst_pos2 + width * channels;
						for (int col = 0; col < right; ++col)
						{
							int dst_index = dst_pos3 + col * channels;
							int src_index = (width - 1) * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index + ch] = src_data[src_index + ch];
							}
						}
					}


					//center
					for (int row = top; row < top + height; ++row)
					{
						int src_pos1 = (row - top) * width * channels;
						int dst_pos1 = row * dst_width * channels;

						//left
						for (int col = 0; col < left; ++col)
						{
							int dst_index = dst_pos1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index + ch] = src_data[src_pos1 + ch];
							}
						}

						//center
						int dst_pos2 = dst_pos1 + left * channels;
						memcpy(dst_data + dst_pos2, src_data + src_pos1, width * channels * sizeof(Dtype));

						//right
						int src_pos2 = src_pos1 + width * channels;
						int dst_pos3 = dst_pos2 + width * channels;
						for (int col = 0; col < right; ++col)
						{
							int dst_index = dst_pos3 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index + ch] = src_data[src_pos2 + ch];
							}
						}
					}


					//bottom
					for (int row = top + height; row < dst_height; ++row)
					{
						int dst_pos1 = row * dst_width * channels;
						int src_pos1 = (height - 1) * width * channels;

						//left
						for (int col = 0; col < left; ++col)
						{
							int dst_index = dst_pos1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index + ch] = src_data[src_pos1 + ch];
							}
						}

						//center
						int dst_pos2 = dst_pos1 + left * channels;
						memcpy(dst_data + dst_pos2, src_data + src_pos1, width * channels * sizeof(Dtype));

						//right
						int dst_pos3 = dst_pos2 + width * channels;
						int src_pos2 = src_pos1 + width * channels;
						for (int col = 0; col < right; ++col)
						{
							int dst_index = dst_pos3 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index + ch] = src_data[src_pos2 + ch];
							}
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support border type.";
					return;
				}
			}
		}

		template <typename Dtype>
		static void cut_border_cpu(const std::shared_ptr<tensor<Dtype>> &src,
			std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right)
		{
			if (top < 0 || bottom < 0 || left < 0 || right < 0)
			{
				LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
				return;
			}

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;

			int dst_height = height - top - bottom;
			int dst_width = width - left - right;
			int dst_offset = dst_height * dst_width;

			if (dst_height == height && dst_width == width)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			if (dst_height <= 0 || dst_width <= 0)
			{
				LOG(ERROR) << "Illegal input size.";
				return;
			}

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				for (int ch = 0; ch < channels; ++ch)
				{
					int src_channel_offset = ch * src_offset;
					int dst_channel_offset = ch * dst_offset;

					for (int row = 0; row < dst_height; ++row)
					{
						int src_index = src_channel_offset + (row + top) * width + left;
						int dst_index = dst_channel_offset + row * dst_width;
						memcpy(dst_data + dst_index, src_data + src_index, dst_width * sizeof(Dtype));
					}
				}
			}
			else
			{
				LOG(WARNING) << "NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				for (int row = 0; row < dst_height; ++row)
				{
					int src_index = ((row + top) * width + left) * channels;
					int dst_index = row * dst_width * channels;
					memcpy(dst_data + dst_index, src_data + src_index, dst_width * channels * sizeof(Dtype));
				}
			}
		}

		template <typename Stype, typename Dtype>
		static void fast_integral_cpu(const std::shared_ptr<tensor<Stype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height + 1, width + 1}, src->device(), src->type()));
				const Stype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();
				memset(dst_data, (Dtype)0, channels * (height + 1) * (width + 1) * sizeof(Dtype));

				// sum of each column
				Dtype *columnSum = new Dtype[width + 1];
				memset(columnSum, Dtype(0), (width + 1) * sizeof(Dtype));

				for (size_t ch = 0; ch < channels; ++ch)
				{
					int src_offset = ch * height * width;
					int dst_offset = ch * (height + 1) * (width + 1);

					// calculate integral of the first nonzero_row(second row) 
					unsigned fist_nonzero_row_index = dst_offset + (width + 1);
					for (int col = 1; col < width + 1; ++col)
					{
						columnSum[col] = (Dtype)src_data[src_offset + col - 1];
						dst_data[fist_nonzero_row_index + col] = (Dtype)src_data[src_offset + col - 1];
						dst_data[fist_nonzero_row_index + col] += dst_data[fist_nonzero_row_index + col - 1];
					}

					for (int row = 2; row < height + 1; ++row)
					{
						int src_row_offset = (row - 1) * width;
						int dst_row_offset = row * (width + 1);

						for (int col = 1; col < width + 1; ++col)
						{
							columnSum[col] += (Dtype)src_data[src_offset + src_row_offset + col - 1];
							dst_data[dst_offset + dst_row_offset + col] = dst_data[dst_offset + dst_row_offset + col - 1] + columnSum[col];
						}
					}
				}

				delete columnSum;
			}
			else
			{
				LOG(WARNING) << "NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, height + 1, width + 1, channels}, src->device(), src->type()));
				const Stype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();
				memset(dst_data, (Dtype)0, channels * (height + 1) * (width + 1) * sizeof(Dtype));

				// sum of each column
				Dtype *columnSum = new Dtype[width + 1];
				memset(columnSum, Dtype(0), (width + 1) * sizeof(Dtype));

				for (size_t ch = 0; ch < channels; ++ch)
				{
					// calculate integral of the first nonzero_row(second row) 
					unsigned fist_nonzero_row_index = (width + 1) * channels + ch;
					for (int col = 1; col < width + 1; ++col)
					{
						columnSum[col] = (Dtype)src_data[(col - 1) * channels + ch];
						dst_data[fist_nonzero_row_index + col * channels] = (Dtype)src_data[(col - 1) * channels + ch];
						dst_data[fist_nonzero_row_index + col * channels] += dst_data[fist_nonzero_row_index + (col - 1) * channels];
					}

					for (int row = 2; row < height + 1; ++row)
					{
						int src_row_offset = (row - 1) * width * channels + ch;
						int dst_row_offset = row * (width + 1) * channels + ch;

						for (int col = 1; col < width + 1; ++col)
						{
							columnSum[col] += (Dtype)src_data[src_row_offset + (col - 1) * channels];
							dst_data[dst_row_offset + col * channels] = dst_data[dst_row_offset + (col - 1) * channels] + columnSum[col];
						}
					}
				}

				delete columnSum;
			}
		}

		template <typename Dtype>
		static void equalize_hist_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			CHECK_EQ(src->num(), 1);
			CHECK_EQ(src->channels(), 1);
			int height = src->height();
			int width = src->width();
			int offset = height * width;
			
			dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device()));
			const Dtype* src_data = src->cpu_data();
			Dtype* dst_data = dst->mutable_cpu_data();

			int gray_value[256] = { 0 };//灰度值
			float probability_distribution[256] = { 0 };//概率密度分布
			float accumulate_probability_distribution[256] = { 0 };//累积概率密度
			int normalized_gray_value[256] = { 0 };//归一化后的灰度值

												   //Count the number of pixels in each grayscale
			for (size_t i = 0; i < offset; i++)
			{
				int value = static_cast<unsigned char>(src_data[i]);
				gray_value[value]++;
			}

			for (int i = 0; i < 256; i++)
			{
				//计算概率密度分布
				probability_distribution[i] = static_cast<float>(gray_value[i]) / offset;

				//计算累积概率密度分布
				if (i > 0)
				{
					accumulate_probability_distribution[i] = accumulate_probability_distribution[i - 1] + probability_distribution[i];
				}
				else
				{
					accumulate_probability_distribution[0] = probability_distribution[0];
				}

				//计算归一化后的像素值
				normalized_gray_value[i] = static_cast<unsigned char>(255 * accumulate_probability_distribution[i] + 0.5);
			}

			for (size_t i = 0; i < offset; i++)
			{
				dst_data[i] = Dtype(normalized_gray_value[static_cast<unsigned char>(src_data[i])]);
			}
		}

		template <typename Dtype>
		static void split_channel_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::vector<std::shared_ptr<tensor<Dtype>>>& dst_vector)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;

			const Dtype* src_data = src->cpu_data();
			dst_vector.clear();

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";

				for (int ch = 0; ch < channels; ++ch)
				{
					int channel_offset = ch * src_offset;
					std::shared_ptr<tensor<Dtype>> temp_ptr;
					temp_ptr.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->type()));
					Dtype* temp_data = temp_ptr->mutable_cpu_data();
					std::memcpy((void*)temp_data, (void*)(src_data + channel_offset), src_offset * sizeof(Dtype));
					dst_vector.push_back(temp_ptr);
				}
			}
			else
			{
				LOG(WARNING) << "NHWC.";

				dst_vector.resize(channels);
				std::vector<Dtype *> ptr_arr(channels);

				for (int ch = 0; ch < channels; ++ch)
				{
					dst_vector.at(ch).reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->type()));
					ptr_arr[ch] = dst_vector.at(ch)->mutable_cpu_data();
				}

				for (int row = 0; row < height; ++row)
				{
					int row_offset = row * width;
					for (int col = 0; col < width; ++col)
					{
						int pos1 = row_offset + col;
						int pos2 = pos1 * channels;
						for (int ch = 0; ch < channels; ++ch)
						{
							ptr_arr[ch][pos1] = src_data[pos2 + ch];
						}
					}
				}
			}
		}

		template <typename Dtype>
		static void merge_channel_cpu(const std::vector<std::shared_ptr<tensor<Dtype>>> &src_vector,  std::shared_ptr<tensor<Dtype>> &dst)
		{
			int num = 0, height, width, device;
			tensorType type;
			for (int i = 0; i < src_vector.size(); ++i)
			{
				num += src_vector.at(i)->num() * src_vector.at(i)->channels();
				if (i == 0) 
				{
					height = src_vector.at(i)->height();
					width = src_vector.at(i)->width();
					device = src_vector.at(i)->device();
					type = src_vector.at(i)->type();
				}
				else 
				{
					if (height != src_vector.at(i)->height() ||
						width != src_vector.at(i)->width() ||
						device != src_vector.at(i)->device() ||
						type != src_vector.at(i)->type())
					{
						LOG(WARNING) << "the element of vector<mat> should have the exact same height/width/device/type.";
						return;
					}
				}
			}

			if (type == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{num, 1, height, width}, device, type));
			}
			else
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{num, height, width, 1}, device, type));
			}
			
			Dtype* dst_data = dst->mutable_cpu_data();
			int offset = height * width;

			unsigned total_length = 0;
			for (int i = 0; i < src_vector.size(); ++i)
			{
				const Dtype* temp_data = src_vector.at(i)->cpu_data();
				unsigned length = src_vector.at(i)->num() * src_vector.at(i)->channels() * offset;
				std::memcpy((void*)(dst_data + total_length), (void*)(temp_data), length * sizeof(Dtype));
				total_length += length;
			}
		}

		template <typename Dtype>
		static void threshold_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary)
		{
			CHECK_EQ(src->num(), 1);
			CHECK_EQ(src->channels(), 1);
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			if (src->type() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->type()));
			}
			else
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->type()));
			}

			Dtype* dst_data = dst->mutable_cpu_data();
			const Dtype *src_data = src->cpu_data();

			switch (type)
			{
			case binary:
				for (int j = 0; j < offset; ++j)
				{
					dst_data[j] = src_data[j] > (Dtype)thresh ? (Dtype)maxval : 0;
				}
				break;

			case binary_inv:
				for (int j = 0; j < offset; ++j)
				{
					dst_data[j] = src_data[j] <= (Dtype)thresh ? (Dtype)maxval : 0;
				}
				break;

			case small_trunc:
				for (int j = 0; j < offset; ++j)
				{
					dst_data[j] = std::max(src_data[j], (Dtype)thresh);
				}
				break;

			case big_trunc:
				for (int j = 0; j < offset; ++j)
				{
					dst_data[j] = std::min(src_data[j], (Dtype)thresh);
				}
				break;

			case small_to_zero:
				for (int j = 0; j < offset; ++j)
				{
					dst_data[j] = src_data[j] >(Dtype)thresh ? src_data[j] : 0;
				}
				break;

			case big_to_zero:
				for (int j = 0; j < offset; ++j)
				{
					dst_data[j] = src_data[j] <= (Dtype)thresh ? src_data[j] : 0;
				}
				break;

			default:
				LOG(ERROR) << "Un-support threshold type.";
				break;
			}
		}

		template <typename Dtype, typename Ptype>
		static void warp_affine_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst,
			const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill = 0, interpolationType type = Bilinear)
		{
			if (src_point.size() != 3 || dst_point.size() != 3)
			{
				LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
				return;
			}

			double a[6 * 6], b[6];
			for (int i = 0; i < 3; i++)
			{
				int j = i * 12;
				int k = i * 12 + 6;
				a[j] = a[k + 3] = (double)dst_point[i].x;
				a[j + 1] = a[k + 4] = (double)dst_point[i].y;
				a[j + 2] = a[k + 5] = 1;
				a[j + 3] = a[j + 4] = a[j + 5] = 0;
				a[k] = a[k + 1] = a[k + 2] = 0;
				b[i * 2] = (double)src_point[i].x;
				b[i * 2 + 1] = (double)src_point[i].y;
			}

			Mat A(6, 6, CV_64F, a), B(6, 1, CV_64F, b);
			Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.ptr());
			cv::solve(A, B, X);
			double *M_data = (double*)malloc(6 * sizeof(double));
			memcpy(M_data, M.data, 6 * sizeof(double));

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				for (int ch = 0; ch < channels; ++ch)
				{
					int channel_offset = ch * offset;

#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						double temp_xf = M_data[1] * row + M_data[2];
						double temp_yf = M_data[4] * row + M_data[5];
						int temp_dst_index = channel_offset + row * width;

						for (int col = 0; col < width; ++col)
						{
							double xf = M_data[0] * col + temp_xf;
							double yf = M_data[3] * col + temp_yf;
							int x = (int)xf;
							int y = (int)yf;
							float xdiff = xf - x;
							float ydiff = yf - y;

							int src_index = channel_offset + y * width + x;
							int dst_index = temp_dst_index + col;

							if (x < 0 || x >= width || y < 0 || y >= height)
							{
								dst_data[dst_index] = (Dtype)fill;
							}
							else
							{
								if (type == Nearest)
								{
									dst_data[dst_index] = src_data[src_index];
								}
								else if (type == Bilinear)
								{
									Dtype A = src_data[src_index];
									Dtype B = src_data[src_index + 1];
									Dtype C = src_data[src_index + width];
									Dtype D = src_data[src_index + width + 1];
									dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
										static_cast<float>(B) * xdiff * (1 - ydiff) +
										static_cast<float>(C) * ydiff * (1 - xdiff) +
										static_cast<float>(D) * xdiff * ydiff);
								}
								else
								{
									LOG(ERROR) << "Un-support interpolation type.";
								}
							}
						}
					}
				}
			}
			else
			{
				LOG(WARNING) << "NHWC.";
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

#pragma omp parallel for
				for (int row = 0; row < height; ++row)
				{
					double temp_xf = M_data[1] * row + M_data[2];
					double temp_yf = M_data[4] * row + M_data[5];
					int dst_pos1 = row * width * channels;

					for (int col = 0; col < width; ++col)
					{
						double xf = M_data[0] * col + temp_xf;
						double yf = M_data[3] * col + temp_yf;
						int x = (int)xf;
						int y = (int)yf;
						float xdiff = xf - x;
						float ydiff = yf - y;

						int src_pos1 = (y * width + x) * channels;
						int dst_pos2 = dst_pos1 + col * channels;

						for (int ch = 0; ch < channels; ++ch)
						{
							if (x < 0 || x >= width || y < 0 || y >= height)
							{
								dst_data[dst_pos2 + ch] = (Dtype)fill;
							}
							else
							{
								if (type == Nearest)
								{
									dst_data[dst_pos2 + ch] = src_data[src_pos1 + ch];
								}
								else if (type == Bilinear)
								{
									int src_pos2 = src_pos1 + width * channels;
									Dtype A = src_data[src_pos1 + ch];
									Dtype B = src_data[src_pos1 + channels + ch];
									Dtype C = src_data[src_pos2 + ch];
									Dtype D = src_data[src_pos2 + channels + ch];
									dst_data[dst_pos2 + ch] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
										static_cast<float>(B) * xdiff * (1 - ydiff) +
										static_cast<float>(C) * ydiff * (1 - xdiff) +
										static_cast<float>(D) * xdiff * ydiff);
								}
								else
								{
									LOG(ERROR) << "Un-support interpolation type.";
								}
							}
						}
					}
				}
			}

			delete M_data;
		}

		template <typename Dtype>
		static void gaussian_blur_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int ksize = 3)
		{
			if (ksize % 2 != 1)
			{
				LOG(WARNING) << "convolution kernel: width and height should be odd.";
				return;
			}

			if (ksize == 1)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}
			 
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			double sigma = ((ksize - 1)*0.5 - 1)*0.3 + 0.8;
			double scale2X = (double)1 / (2 * sigma * sigma);
			double prefix = (double)1 / sqrt(2 * PI * sigma * sigma);

			std::vector<double> convolution_kernel(ksize);
			double sum = 0;
			int half = (ksize - 1) * 0.5;
			for (int col = 0; col < ksize; ++col)
			{
				double dx = col - half;
				double distance = dx * dx;
				convolution_kernel[col] = prefix * std::exp(-1 * distance * scale2X);
				sum += convolution_kernel[col];
			}

			for (int col = 0; col < ksize; ++col)
			{
				convolution_kernel[col] /= sum;
			}
			
			const Dtype* src_data = src->cpu_data();

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();

				std::shared_ptr<tensor<Dtype>> temp;
				temp.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
				Dtype* temp_data = temp->mutable_cpu_data();

				//horizontal
				for (int ch = 0; ch < channels; ++ch)
				{
					int channel_offset = ch * offset;

#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						int index = channel_offset + row * width;
						for (int col = 0; col < width; ++col)
						{
							double sum2 = 0;
							for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
							{
								if (col + kernel_col < 0 || col + kernel_col >= width)
								{
									continue;
								}
								else
								{
									sum2 += convolution_kernel[kernel_col + half] * src_data[index + (col + kernel_col)];
								}
							}
							temp_data[index + col] = (Dtype)sum2;
						}
					}
				}

				//vertical
				for (int ch = 0; ch < channels; ++ch)
				{
					int channel_offset = ch * offset;

#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						int index = channel_offset + row * width;
						for (int col = 0; col < width; ++col)
						{
							int pos = index + col;
							double sum2 = 0;
							for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
							{
								if (row + kernel_row < 0 || row + kernel_row >= height)
								{
									continue;
								}
								else
								{
									sum2 += convolution_kernel[kernel_row + half] * temp_data[pos + kernel_row * width];
								}
							}
							dst_data[pos] = (Dtype)sum2;
						}
					}
				}
			}
			else
			{
				LOG(WARNING) << "NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();

				std::shared_ptr<tensor<Dtype>> temp;
				temp.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
				Dtype* temp_data = temp->mutable_cpu_data();

				//horizontal
#pragma omp parallel for
				for (int row = 0; row < height; ++row)
				{
					int index = row * width * channels;
					for (int col = 0; col < width; ++col)
					{
						for (int ch = 0; ch < channels; ++ch)
						{
							double sum2 = 0;
							for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
							{
								if (col + kernel_col < 0 || col + kernel_col >= width)
								{
									continue;
								}
								else
								{
									sum2 += convolution_kernel[kernel_col + half] * src_data[index + (col + kernel_col) * channels + ch];
								}
							}
							temp_data[index + col * channels + ch] = (Dtype)sum2;
						}
					}
				}


				//vertical
#pragma omp parallel for
				for (int row = 0; row < height; ++row)
				{
					int index = row * width * channels;
					for (int col = 0; col < width; ++col)
					{
						int pos = index + col * channels;
						for (int ch = 0; ch < channels; ++ch)
						{
							double sum2 = 0;
							for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
							{
								if (row + kernel_row < 0 || row + kernel_row >= height)
								{
									continue;
								}
								else
								{
									sum2 += convolution_kernel[kernel_row + half] * temp_data[pos + kernel_row * width * channels + ch];
								}
							}
							dst_data[pos + ch] = (Dtype)sum2;
						}
					}
				}
			}
		}

		template <typename Dtype>
		static void sobel_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int dx = 1, int dy = 1)
		{
			if (dx == 0 && dy == 0)
			{
				LOG(WARNING) << "dx, dy cannot be zero simultaneously.";
				return;
			}

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				for (int ch = 0; ch < channels; ++ch)
				{
					int channel_offset = ch * offset;

#pragma omp parallel for
					for (int row = 1; row < height - 1; ++row)
					{
						int index = channel_offset + row * width;
						for (int col = 1; col < width - 1; ++col)
						{
							int pos = index + col;
							int posAdd = pos + width;
							int posSub = pos - width;

							int sumx = 0, sumy = 0;

							if (dx > 0)
							{
								sumx = src_data[posSub + 1] + 2 * src_data[pos + 1] + src_data[posAdd + 1]
									- src_data[posSub - 1] - 2 * src_data[pos - 1] - src_data[posAdd - 1];
							}

							if (dy > 0)
							{
								sumy = src_data[posSub - 1] + 2 * src_data[posSub] + src_data[posSub + 1]
									- src_data[posAdd - 1] - 2 * src_data[posAdd] - src_data[posAdd + 1];
							}

							int total = abs(sumx) + abs(sumy);
							if (sumx != 0 && sumy != 0)
							{
								total = 0.35 * total;
							}
							else
							{
								total = 0.6 * total;
							}

							if (total > 255)
							{
								total = 255;
							}

							dst_data[pos] = total;
						}
					}
				}
			}
			else
			{
				LOG(WARNING) << "NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

#pragma omp parallel for
				for (int row = 1; row < height - 1; ++row)
				{
					int index = row * width * channels;
					for (int col = 1; col < width - 1; ++col)
					{
						int pos = index + col * channels;
						int posAdd = pos + width * channels;
						int posSub = pos - width * channels;									

						for (int ch = 0; ch < channels; ++ch)
						{
							int sumx = 0, sumy = 0;

							if (dx > 0)
							{
								sumx = src_data[posSub + channels + ch] + 2 * src_data[pos + channels + ch] + src_data[posAdd + channels + ch]
									- src_data[posSub - channels + ch] - 2 * src_data[pos - channels + ch] - src_data[posAdd - channels + ch];
							}

							if (dy > 0)
							{
								sumy = src_data[posSub - channels + ch] + 2 * src_data[posSub + ch] + src_data[posSub + channels + ch]
									- src_data[posAdd - channels + ch] - 2 * src_data[posAdd + ch] - src_data[posAdd + channels + ch];
							}

							int total = abs(sumx) + abs(sumy);
							if (sumx != 0 && sumy != 0)
							{
								total = 0.35 * total;
							}
							else
							{
								total = 0.6 * total;
							}

							if (total > 255)
							{
								total = 255;
							}

							dst_data[pos + ch] = total;
						}
					}
				}
			}
		}

		template <typename Dtype>
		static void morph_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, morphType type = Dilate, int ksize = 3)
		{
			if (ksize % 2 != 1)
			{
				LOG(WARNING) << "ksize should be odd.";
				return;
			}

			if (ksize == 1)
			{
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				LOG(WARNING) << "Just copy from the source.";
				return;
			}

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;
			int half = (ksize - 1) * 0.5;

			if (src->type() == NCHW)
			{
				LOG(WARNING) << "NCHW.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				if (type == morphType::Dilate)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

#pragma omp parallel for
						for (int row = 0; row < height; ++row)
						{
							int index = channel_offset + row * width;
							for (int col = 0; col < width; ++col)
							{
								int pos1 = index + col;
								int max = -99999;
								for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
								{
									if (row + kernel_row < 0 || row + kernel_row >= height)
									{
										continue;
									}

									int pos2 = pos1 + kernel_row * width;
									for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
									{
										if (col + kernel_col < 0 || col + kernel_col >= width)
										{
											continue;
										}

										int pos3 = pos2 + kernel_col;
										if (src_data[pos3] > max)
										{
											max = src_data[pos3];
										}
									}
								}
								dst_data[pos1] = max;
							}
						}
					}
				}
				else if (type == morphType::Erode)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

#pragma omp parallel for
						for (int row = 0; row < height; ++row)
						{
							int index = channel_offset + row * width;
							for (int col = 0; col < width; ++col)
							{
								int pos1 = index + col;
								int min = 99999;
								for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
								{
									if (row + kernel_row < 0 || row + kernel_row >= height)
									{
										continue;
									}

									int pos2 = pos1 + kernel_row * width;
									for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
									{
										if (col + kernel_col < 0 || col + kernel_col >= width)
										{
											continue;
										}

										int pos3 = pos2 + kernel_col;
										if (src_data[pos3] < min)
										{
											min = src_data[pos3];
										}
									}
								}
								dst_data[pos1] = min;
							}
						}
					}
				}
				else
				{
					LOG(WARNING) << "unsupported type.";
				}
			}
			else
			{
				LOG(WARNING) << "NHWC.";

				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				if (type == morphType::Dilate)
				{
#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						int index = row * width * channels;
						for (int col = 0; col < width; ++col)
						{
							int pos1 = index + col * channels;

							for (int ch = 0; ch < channels; ++ch)
							{
								int max = -99999;
								for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
								{
									if (row + kernel_row < 0 || row + kernel_row >= height)
									{
										continue;
									}

									int pos2 = pos1 + kernel_row * width * channels + ch;
									for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
									{
										if (col + kernel_col < 0 || col + kernel_col >= width)
										{
											continue;
										}

										int pos3 = pos2 + kernel_col * channels;
										if (src_data[pos3] > max)
										{
											max = src_data[pos3];
										}
									}
								}
								dst_data[pos1 + ch] = max;
							}
						}
					}
				}
				else if (type == morphType::Erode)
				{
#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						int index = row * width * channels;
						for (int col = 0; col < width; ++col)
						{
							int pos1 = index + col * channels;
							
							for (int ch = 0; ch < channels; ++ch)
							{
								int min = 99999;
								for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
								{
									if (row + kernel_row < 0 || row + kernel_row >= height)
									{
										continue;
									}

									int pos2 = pos1 + kernel_row * width * channels + ch;
									for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
									{
										if (col + kernel_col < 0 || col + kernel_col >= width)
										{
											continue;
										}

										int pos3 = pos2 + kernel_col * channels;
										if (src_data[pos3] < min)
										{
											min = src_data[pos3];
										}
									}
								}
								dst_data[pos1 + ch] = min;
							}
						}
					}			
				}
				else
				{
					LOG(WARNING) << "unsupported type.";
				}
			}
		}





		template <typename Dtype>
		static void lbp_feature_cpu(const tensor<Dtype>* src, tensor<Dtype>* dst, lbpType type)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			const Dtype* src_data = src->cpu_data();
			Dtype* dst_data = dst->mutable_cpu_data();
			switch (type)
			{
			case excalibur::Native:
				lbp_feature_cpu_native(src_data, height, width, channels, dst_data);
				break;
			case excalibur::RI:
				NOT_IMPLEMENTED;
				break;
			case excalibur::U2:
				NOT_IMPLEMENTED;
				break;
			case excalibur::RIU2:
				NOT_IMPLEMENTED;
				break;
			case excalibur::HF:
				NOT_IMPLEMENTED;
				break;
			case excalibur::LTP:
				NOT_IMPLEMENTED;
				break;
			default:
				LOG(ERROR) << "Un-supported LBP type.";
				break;
			}
			showimage(dst);
		}

		template <typename Dtype>
		static void mblbp_feature_cpu(const tensor<Dtype>* src, tensor<Dtype>* dst, int block_h, 
			int block_w, int stride_h, int stride_w)
		{
			CHECK_EQ(src->num(), 1);
			CHECK_GE(block_h, 1);
			CHECK_GE(block_w, 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			CHECK_GE(height, 3 * block_h);
			CHECK_GE(width, 3 * block_w);
			const Dtype* src_data = src->cpu_data();
			Dtype* dst_data = dst->mutable_cpu_data();
			Dtype* integral_data = new Dtype[channels * (height + 1) * (width + 1)];
			fast_integral_cpu(src_data, height, width, channels, integral_data);
			int dst_height = dst->height();
			int dst_width = dst->width();
			for (size_t c = 0; c < channels; c++)
			{
				int src_offset = c * height * width;
				int dst_offset = c * dst_height * dst_width;
				int integral_offset = c * (height + 1) * (width + 1);
				for (size_t h = 0; h < height - 3 * block_h + 1; h += stride_h)
				{
					int dst_sub_offset = h / stride_h * dst_width;
					for (size_t w = 0; w < width - 3 * block_w + 1; w += stride_w)
					{
						Dtype block_values[9];
						for (size_t i = 0; i < 9; i++)
						{
							int x1 = w + (i % 3) * block_w;
							int y1 = h + (i / 3) * block_h;
							int x2 = x1 + block_w;
							int y2 = y1 + block_h;
							float A = (float)integral_data[integral_offset + y1 * (height + 1) + x1];
							float B = (float)integral_data[integral_offset + y1 * (height + 1) + x2];
							float C = (float)integral_data[integral_offset + y2 * (height + 1) + x1];
							float D = (float)integral_data[integral_offset + y2 * (height + 1) + x2];
							block_values[i] = Dtype(D - B - C + A);
						}
						unsigned char code = 0;
						Dtype center = block_values[4];
						code |= (block_values[0] >= center) << 0;
						code |= (block_values[1] >= center) << 1;
						code |= (block_values[2] >= center) << 2;
						code |= (block_values[3] >= center) << 7;
						code |= (block_values[5] >= center) << 3;
						code |= (block_values[6] >= center) << 6;
						code |= (block_values[7] >= center) << 5;
						code |= (block_values[8] >= center) << 4;
						dst_data[dst_offset + dst_sub_offset + w / stride_w] = Dtype(code);
					}
				}
			}
			delete integral_data;
		}

		template <typename Dtype>
		static void preprocess_tensors_cpu(tensor<Dtype>* dst)
		{
			int num = dst->num();
			int channel = dst->channels();
			CHECK_EQ(channel, 3);
			int height = dst->height();
			int width = dst->width();
			Dtype* dst_data = dst->mutable_cpu_data();
			float means[] = { 104.f, 117.0f, 124.f };
			float var = 0.0078125f;
			for (size_t n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (size_t c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (size_t h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (size_t w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								Dtype((dst_data[offset + sub_offset + subsub_offset + w] - means[c]) * var);
						}
					}
				}
			}
		}

		template <typename Dtype, typename Rtype>
		static void safty_cut_cpu(const tensor<Dtype>* src, tensor<Dtype>* dst, rectangle<Rtype>* rect)
		{
			if (rect->x >= 0 && rect->y >=0 && (rect->x + rect->w <= src->width()) && (rect->y + rect->h <= src->height()))
			{
				copy_cut_border_cpu(src, dst, rect->y, (src->height() - rect->y - rect->h), rect->x, (src->width() - rect->x - rect->w));
			}
			else
			{
				int top = std::max(0, -1 * rect->x);
				int bottom = std::max(rect->y + rect->h - src->height(), 0);
				int left = std::max(0, -1 * rect->y);
				int right = std::max(rect->x + rect->w - src->width(), 0);
				tensor<Dtype>* temp = new tensor<Dtype>(
					std::vector<int>{src->num(), src->channels(), src->height() + top + bottom, src->width() + left + right}, 
					src->device());
				copy_make_border_cpu(src, temp, top, bottom, left, right, Border_Constant, Dtype(0));
				copy_cut_border_cpu(temp, dst, rect->y + top, temp->height() - rect->y - rect->h - top, rect->x + left, temp->width() - rect->x - rect->w - left);
				delete temp;
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
		static void convert2mat(const tensor<Dtype>* src, cv::Mat& dst)
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
						//std::cout << dst_data[w*channel + c] << " ";
					}
				}
				//std::cout << std::endl;
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

		template <typename Dtype>
		static void convert2tensor(const cv::Mat src, tensor<Dtype>* dst)
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
			dst = new tensor<Dtype>(std::vector<int>{1, channel, height, width}, -1);
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

			if (type == Border_Constant)
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
			else if (type == Border_Replicate)
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
		static void lbp_feature_cpu_native(const Dtype* src_data, int height, int width, int channels, Dtype* dst_data)
		{
			for (size_t c = 0; c < channels; c++)
			{
				int src_c_offset = c * height * width;
				int dst_c_offset = c * (height - 2) * (width - 2);
				for (size_t h = 1; h < height - 1; h++)
				{
					int offset = src_c_offset + h * width;
					int offset_plus = offset + width;
					int offset_minus = offset - width;
					for (size_t w = 1; w < width - 1; w++)
					{
						Dtype center = src_data[offset + w];
						unsigned char code = 0;
						code |= (src_data[offset_minus + w - 1] >= center) << 7;
						code |= (src_data[offset_minus + w - 0] >= center) << 6;
						code |= (src_data[offset_minus + w + 1] >= center) << 5;
						code |= (src_data[offset + w + 1] >= center) << 4;
						code |= (src_data[offset_plus + w + 1] >= center) << 3;
						code |= (src_data[offset_plus + w + 0] >= center) << 2;
						code |= (src_data[offset_plus + w - 1] >= center) << 1;
						code |= (src_data[offset + w - 1] >= center) << 0;
						dst_data[dst_c_offset + (h - 1) * (width - 2) + w - 1] = static_cast<Dtype>(code);
					}
				}
			}
		}

#ifdef USE_OPENCV
		template <typename Dtype>
		static void showimage(tensor<Dtype>* dst)
		{
			cv::Mat mat;
			convert2mat(dst, mat);
			cv::Mat showmat;
			mat.convertTo(showmat, CV_8U);
			cv::imshow("test", showmat);
			cv::waitKey();
		}

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
