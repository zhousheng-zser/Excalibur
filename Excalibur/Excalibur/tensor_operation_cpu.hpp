#ifndef _TENSOR_OPERATION_CPU_HPP_
#define _TENSOR_OPERATION_CPU_HPP_

#include <glasssix\tensor.hpp>
#include "tensor_utils.hpp"
#include "math_functions.hpp"
#include <algorithm>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <glasssix\timer.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PI 3.141592


namespace glasssix
{
	namespace excalibur
	{
		class tensor_operation_cpu
		{
		public:
			tensor_operation_cpu() {};
			~tensor_operation_cpu() {};

#define ICV_WARP_SHIFT          10
#define ICV_WARP_SHIFT2         15
#define ICV_SHIFT_DIFF          (ICV_WARP_SHIFT2-ICV_WARP_SHIFT)
#define ICV_WARP_MASK           ((1 << ICV_WARP_SHIFT) - 1)

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

#define ICV_WARP_MUL_ONE_8U(x)  ((x) << ICV_WARP_SHIFT)
#define ICV_WARP_DESCALE_8U(x)  CV_DESCALE((x), ICV_WARP_SHIFT*2)
#define CV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))

			typedef struct CvResizeAlpha
			{
				int idx;
				int ialpha;

			}CvResizeAlpha;


#ifdef USE_OPENCV

			/// <summary>
			/// tensor转换为mat
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">转换得到的mat数据</param>
			template <typename Dtype>
			static void tensor2mat(const std::shared_ptr<tensor<Dtype>> &src, cv::Mat& dst)
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

				if (src->order() == NCHW)
				{
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
				else if(src->order() == NHWC)
				{
					std::memcpy(dst.data, src->cpu_data(), height * width * channel * sizeof(Dtype));
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// tensor转换为mat
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">转换得到的mat数据</param>
			template <typename Dtype>
			static void tensor2mat(const tensor<Dtype>& src, cv::Mat& dst)
			{
				CHECK_EQ(src.num(), 1);
				int channel = src.channels();
				if (channel>4)
				{
					LOG(ERROR) << "Too many channels.";
					return;
				}
				int width = src.width();
				int height = src.height();
				int type = get_cv_type<Dtype>();
				if (type<0)
				{
					LOG(ERROR) << "Un-support data type.";
					return;
				}
				dst = cv::Mat(height, width, CV_MAKETYPE(type, channel));

				if (src.order() == NCHW)
				{
					const Dtype* src_data = src.cpu_data();
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
				else if (src.order() == NHWC)
				{
					std::memcpy(dst.data, src.cpu_data(), height * width * channel * sizeof(Dtype));
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// mat转换为tensor
			/// </summary>
			/// <param name="src">mat数据</param>
			/// <param name="dst">转换得到的tensor</param>
			/// <param name="order">指定tensor类型，默认为NHWC</param>
			template <typename Dtype>
			static void mat2tensor(const cv::Mat &src, std::shared_ptr<tensor<Dtype>>& dst, orderType order = NHWC)
			{
				int channels = src.channels();
				int width = src.cols;
				int height = src.rows;
				if (src.data == NULL)
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

				if (order == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, -1, NCHW));
					Dtype* dst_data = dst->mutable_cpu_data();
					int dst_offset = width * height;
					int* c_dst_offset = new int[channels];
					for (int c = 0; c < channels; c++)
					{
						c_dst_offset[c] = c * dst_offset;
					}
					for (int c = 0; c < channels; c++)
					{
						for (int h = 0; h < height; h++)
						{
							const Dtype* src_data = src.ptr<Dtype>(h);
							int dst_sub_offset = h * width;
							for (int w = 0; w < width; w++)
							{
								dst_data[c_dst_offset[c] + dst_sub_offset + w] =
									src_data[w * channels + c];
							}
						}
					}
					delete c_dst_offset;
				}
				else if (order == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, -1, NHWC));
					Dtype* dst_data = dst->mutable_cpu_data();
					memcpy(dst_data, src.data, height * width * channels * sizeof(Dtype));
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// mat转换为tensor
			/// </summary>
			/// <param name="src">mat数据</param>
			/// <param name="dst">转换得到的tensor</param>
			/// <param name="order">指定tensor类型，默认为NHWC</param>
			template <typename Dtype>
			static void mat2tensor(const cv::Mat &src, tensor<Dtype>& dst, orderType order = NHWC)
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

				if (order == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channel, height, width}, -1, NCHW);
					Dtype* dst_data = dst.mutable_cpu_data();
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
				else if (order == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, channel}, -1, NHWC);
					Dtype* dst_data = dst.mutable_cpu_data();
					memcpy(dst_data, src.data, height * width * channel * sizeof(Dtype));
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}


#endif



			/// <summary>
			/// tensor类型转换
			/// </summary>
			/// <param name="src">源tensor</param>
			/// <param name="dst">转换后的tensor</param>
			template <typename DtypeSRC, typename DtypeDST>
			static void type_converter_cpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst)
			{
				const DtypeSRC* src_data = src->cpu_data();
				DtypeDST* dst_data = dst->mutable_cpu_data();
				for (size_t i = 0; i < src->count(); i++)
				{
					dst_data[i] = DtypeDST(src_data[i]);
				}
			}



			/// <summary>
			/// tensor类型转换
			/// </summary>
			/// <param name="src">源tensor</param>
			/// <param name="dst">转换后的tensor</param>
			template <typename DtypeSRC, typename DtypeDST>
			static void type_converter_cpu(const tensor<DtypeSRC>* src, tensor<DtypeDST>* dst)
			{
				const DtypeSRC* src_data = src->cpu_data();
				DtypeDST* dst_data = dst->mutable_cpu_data();
				for (size_t i = 0; i < src->count(); i++)
				{
					dst_data[i] = DtypeDST(src_data[i]);
				}
			}



			/// <summary>
			/// 将NCHW排列的tensor转换为NHWC排列
			/// </summary>
			/// <param name="src">NCHW排列的tensor</param>
			/// <param name="dst">NHWC排列的tensor</param>
			template <typename Dtype>
			static void nchw2nhwc(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst)
			{
				CHECK_EQ(src->order(), NCHW);
				int num = src->num();
				int height = src->height();
				int width = src->width();
				int channels = src->channels();
				int offset = height * width;

				dst.reset(new tensor<Dtype>(std::vector<int>{num, height, width, channels}, src->device(), NHWC));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				for (int n = 0; n < num; n++)
				{
					int n_offset = n * channels * offset;
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;
						for (int row = 0; row < height; ++row)
						{
							int row_offset = row * width;
							for (int col = 0; col < width; ++col)
							{
								dst_data[n_offset + (row_offset + col) * channels + ch] = src_data[n_offset + channel_offset + row_offset + col];
							}
						}
					}
				}

			}



			/// <summary>
			/// 将NCHW排列的tensor转换为NHWC排列
			/// </summary>
			/// <param name="src">NCHW排列的tensor</param>
			/// <param name="dst">NHWC排列的tensor</param>
			template <typename Dtype>
			static void nchw2nhwc(const tensor<Dtype> &src, tensor<Dtype> &dst)
			{
				CHECK_EQ(src.order(), NCHW);
				int num = src.num();
				int height = src.height();
				int width = src.width();
				int channels = src.channels();
				int offset = height * width;

				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), NHWC);
				Dtype* dst_data = dst.mutable_cpu_data();
				const Dtype* src_data = src.cpu_data();

				for (int n = 0; n < num; n++)
				{
					int n_offset = n * channels * offset;
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;
						for (int row = 0; row < height; ++row)
						{
							int row_offset = row * width;
							for (int col = 0; col < width; ++col)
							{
								dst_data[n_offset + (row_offset + col) * channels + ch] = src_data[n_offset + channel_offset + row_offset + col];
							}
						}
					}
				}
			}



			/// <summary>
			/// 将NHWC排列的tensor转换为NCHW排列
			/// </summary>
			/// <param name="src">NHWC排列的tensor</param>
			/// <param name="dst">NCHW排列的tensor</param>
			template <typename Dtype>
			static void nhwc2nchw(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst)
			{
				CHECK_EQ(src->order(), NHWC);
				int num = src->num();
				int height = src->height();
				int width = src->width();
				int channels = src->channels();
				int offset = height * width;

				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), NCHW));
				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				for (int n = 0; n < num; n++)
				{
					int n_offset = n * channels * offset;
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;
						for (int row = 0; row < height; ++row)
						{
							int row_offset = row * width;
							for (int col = 0; col < width; ++col)
							{
								dst_data[n_offset + channel_offset + row_offset + col] = src_data[n_offset + (row_offset + col) * channels + ch];
							}
						}
					}
				}
			}



			/// <summary>
			/// 将NHWC排列的tensor转换为NCHW排列
			/// </summary>
			/// <param name="src">NHWC排列的tensor</param>
			/// <param name="dst">NCHW排列的tensor</param>
			template <typename Dtype>
			static void nhwc2nchw(const tensor<Dtype> &src, tensor<Dtype> &dst)
			{
				CHECK_EQ(src.order(), NHWC);
				int num = src.num();
				int height = src.height();
				int width = src.width();
				int channels = src.channels();
				int offset = height * width;

				dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), NCHW);
				Dtype* dst_data = dst.mutable_cpu_data();
				const Dtype* src_data = src.cpu_data();

				for (int n = 0; n < num; n++)
				{
					int n_offset = n * channels * offset;
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;
						for (int row = 0; row < height; ++row)
						{
							int row_offset = row * width;
							for (int col = 0; col < width; ++col)
							{
								dst_data[n_offset + channel_offset + row_offset + col] = src_data[n_offset + (row_offset + col) * channels + ch];
							}
						}
					}
				}
			}




			/// <summary>
			/// 根据新尺寸变换图像大小，如果图像为单通道灰度图，使用gray_resize_cpu替代
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">尺寸变换后的tensor</param>
			/// <param name="dst_height">新的图像高度</param>
			/// <param name="dst_width">新的图像宽度</param>
			/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
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
				unsigned maxIndex = height * width * channels - 1;

				if (dst_width == width && dst_height == height)
				{
					LOG(WARNING) << "Just copy from the source.";
					dst = std::make_shared<tensor<Dtype>>(src->clone());
					return;
				}

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));

					Dtype* dst_data = dst->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();
					int src_offset = height * width;
					int dst_offset = dst_height * dst_width;

					void* temp_buf = 0;
					int scale_x, scale_y;
					int sx, sy, dx, dy;
					int xmax = dst_width, buf_size;
					int *buf0, *buf1;
					CvResizeAlpha *xofs, *yofs;
					int fx_1024x, fy_1024x;

					scale_x = ((width << ICV_WARP_SHIFT2) + dst_width / 2) / dst_width;
					scale_y = ((height << ICV_WARP_SHIFT2) + dst_height / 2) / dst_height;

					buf_size = dst_width * 2 * sizeof(int) + (dst_width + dst_height) * sizeof(CvResizeAlpha);
					temp_buf = buf0 = (int*)malloc(buf_size);
					buf1 = buf0 + dst_width;
					xofs = (CvResizeAlpha*)(buf1 + dst_width);
					yofs = xofs + dst_width;

					for (dx = 0; dx < dst_width; dx++)
					{
						fx_1024x = ((dx * 2 + 1)*scale_x - (1 << ICV_WARP_SHIFT2)) / 2;
						sx = (fx_1024x >> ICV_WARP_SHIFT2);
						fx_1024x = ((fx_1024x - (sx << ICV_WARP_SHIFT2)) >> ICV_SHIFT_DIFF);

						if (sx < 0)
							sx = 0, fx_1024x = 0;

						if (sx >= width - 1)
						{
							fx_1024x = 0, sx = width - 1;
							if (xmax >= dst_width)
								xmax = dx;
						}

						xofs[dx].idx = sx;
						xofs[dx].ialpha = fx_1024x;
					}

					for (dy = 0; dy < dst_height; dy++)
					{
						fy_1024x = ((dy * 2 + 1)*scale_y - (1 << ICV_WARP_SHIFT2)) / 2;
						sy = (fy_1024x >> ICV_WARP_SHIFT2);
						fy_1024x = ((fy_1024x - (sy << ICV_WARP_SHIFT2)) >> ICV_SHIFT_DIFF);

						if (sy < 0)
							sy = 0, fy_1024x = 0;

						yofs[dy].idx = sy;
						yofs[dy].ialpha = fy_1024x;
					}

					for (int ch = 0; ch < channels; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						int dst_channel_offset = ch * dst_offset;

						icvResize_Bilinear_8u_C1(&src_data[src_channel_offset], width * sizeof(Dtype), width, height, &dst_data[dst_channel_offset],
							dst_width * sizeof(Dtype), dst_width, dst_height, xmax, xofs, yofs, buf0, buf1);
					}

					free(temp_buf);
				}
				else if (src->order() == NHWC)
				{
					float width_ratio = (float)width / dst_width;
					float height_ratio = (float)height / dst_height;
					float beta = 0.5f;

					dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

#pragma omp parallel for
					for (int row = 0; row < dst_height; ++row)
					{
						float yf = row * height_ratio + beta;
						int y = (int)yf;
						float ydiff = yf - y;

						int src_pos1 = y * width * channels;
						int dst_pos1 = row * dst_width * channels;

						for (int col = 0; col < dst_width; ++col)
						{
							float xf = col * width_ratio + beta;
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
									unsigned indexA = std::min(unsigned(src_pos3), maxIndex);
									unsigned indexB = std::min(unsigned(src_pos3 + channels), maxIndex);
									unsigned indexC = std::min(unsigned(src_pos3 + width * channels), maxIndex);
									unsigned indexD = std::min(unsigned(src_pos3 + (width + 1) * channels), maxIndex);
									Dtype A = src_data[indexA];
									Dtype B = src_data[indexB];
									Dtype C = src_data[indexC];
									Dtype D = src_data[indexD];

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 根据新尺寸变换图像大小，如果图像为单通道灰度图，使用gray_resize_cpu替代
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">尺寸变换后的tensor</param>
			/// <param name="dst_height">新的图像高度</param>
			/// <param name="dst_width">新的图像宽度</param>
			/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
			template <typename Dtype>
			static void resize_cpu(const tensor<Dtype> &src, tensor<Dtype> &dst,
				int dst_height, int dst_width, interpolationType type = Bilinear)
			{
				if (dst_height * dst_width <= 0)
				{
					LOG(ERROR) << "Illegal input size.";
					return;
				}

				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				unsigned maxIndex = height * width * channels - 1;

				if (dst_width == width && dst_height == height)
				{
					LOG(WARNING) << "Just copy from the source.";
					dst = src.clone();
					return;
				}

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.order());

					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();
					int src_offset = height * width;
					int dst_offset = dst_height * dst_width;

					void* temp_buf = 0;
					int scale_x, scale_y;
					int sx, sy, dx, dy;
					int xmax = dst_width, buf_size;
					int *buf0, *buf1;
					CvResizeAlpha *xofs, *yofs;
					int fx_1024x, fy_1024x;

					scale_x = ((width << ICV_WARP_SHIFT2) + dst_width / 2) / dst_width;
					scale_y = ((height << ICV_WARP_SHIFT2) + dst_height / 2) / dst_height;

					buf_size = dst_width * 2 * sizeof(int) + (dst_width + dst_height) * sizeof(CvResizeAlpha);
					temp_buf = buf0 = (int*)malloc(buf_size);
					buf1 = buf0 + dst_width;
					xofs = (CvResizeAlpha*)(buf1 + dst_width);
					yofs = xofs + dst_width;

					for (dx = 0; dx < dst_width; dx++)
					{
						fx_1024x = ((dx * 2 + 1)*scale_x - (1 << ICV_WARP_SHIFT2)) / 2;
						sx = (fx_1024x >> ICV_WARP_SHIFT2);
						fx_1024x = ((fx_1024x - (sx << ICV_WARP_SHIFT2)) >> ICV_SHIFT_DIFF);

						if (sx < 0)
							sx = 0, fx_1024x = 0;

						if (sx >= width - 1)
						{
							fx_1024x = 0, sx = width - 1;
							if (xmax >= dst_width)
								xmax = dx;
						}

						xofs[dx].idx = sx;
						xofs[dx].ialpha = fx_1024x;
					}

					for (dy = 0; dy < dst_height; dy++)
					{
						fy_1024x = ((dy * 2 + 1)*scale_y - (1 << ICV_WARP_SHIFT2)) / 2;
						sy = (fy_1024x >> ICV_WARP_SHIFT2);
						fy_1024x = ((fy_1024x - (sy << ICV_WARP_SHIFT2)) >> ICV_SHIFT_DIFF);

						if (sy < 0)
							sy = 0, fy_1024x = 0;

						yofs[dy].idx = sy;
						yofs[dy].ialpha = fy_1024x;
					}

					for (int ch = 0; ch < channels; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						int dst_channel_offset = ch * dst_offset;

						icvResize_Bilinear_8u_C1(&src_data[src_channel_offset], width * sizeof(Dtype), width, height, &dst_data[dst_channel_offset],
							dst_width * sizeof(Dtype), dst_width, dst_height, xmax, xofs, yofs, buf0, buf1);
					}

					free(temp_buf);
				}
				else if (src.order() == NHWC)
				{
					float width_ratio = (float)width / dst_width;
					float height_ratio = (float)height / dst_height;
					float beta = 0.5f;

					dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

#pragma omp parallel for
					for (int row = 0; row < dst_height; ++row)
					{
						float yf = row * height_ratio + beta;
						int y = (int)yf;
						float ydiff = yf - y;

						int src_pos1 = y * width * channels;
						int dst_pos1 = row * dst_width * channels;

						for (int col = 0; col < dst_width; ++col)
						{
							float xf = col * width_ratio + beta;
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
									unsigned indexA = std::min(unsigned(src_pos3), maxIndex);
									unsigned indexB = std::min(unsigned(src_pos3 + channels), maxIndex);
									unsigned indexC = std::min(unsigned(src_pos3 + width * channels), maxIndex);
									unsigned indexD = std::min(unsigned(src_pos3 + (width + 1) * channels), maxIndex);
									Dtype A = src_data[indexA];
									Dtype B = src_data[indexB];
									Dtype C = src_data[indexC];
									Dtype D = src_data[indexD];

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 图像绕中心点旋转某个角度，旋转后图像的宽度和高度随之改变
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">旋转变换后的tensor</param>
			/// <param name="theta">旋转角度，逆时针方向为正</param>
			/// <param name="dst_height">旋转后新的图像高度</param>
			/// <param name="dst_width">旋转后新的图像宽度</param>
			/// <param name="fill">空白区域填充值，默认为0</param>
			/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
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
				unsigned maxIndex = height * width * channels - 1;

				float rad = -1 * theta * (PI / 180);//逆时针方向为正
				float cosa = cos(rad);
				float sina = sin(rad);

				dst_width = (int)(width * abs(cosa) + height * abs(sina));
				dst_height = (int)(width * abs(sina) + height * abs(cosa));

				float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
				float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
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
										unsigned indexA = std::min(unsigned(src_pos1), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos1 + 1), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos1 + width), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos1 + width + 1), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
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
										unsigned indexA = std::min(unsigned(src_pos2), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos2 + channels), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos2 + width * channels), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 图像绕中心点旋转某个角度，旋转后图像的宽度和高度随之改变
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">旋转变换后的tensor</param>
			/// <param name="theta">旋转角度，逆时针方向为正</param>
			/// <param name="dst_height">旋转后新的图像高度</param>
			/// <param name="dst_width">旋转后新的图像宽度</param>
			/// <param name="fill">空白区域填充值，默认为0</param>
			/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
			template <typename Dtype>
			static void rotate_with_center_cpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
				float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear)
			{
				if (fabs(theta) <= 1e-6)
				{
					LOG(WARNING) << "Just copy from the source.";
					dst = src.clone();
					return;
				}

				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				unsigned maxIndex = height * width * channels - 1;

				float rad = -1 * theta * (PI / 180);//逆时针方向为正
				float cosa = cos(rad);
				float sina = sin(rad);

				dst_width = (int)(width * abs(cosa) + height * abs(sina));
				dst_height = (int)(width * abs(sina) + height * abs(cosa));

				float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
				float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
										unsigned indexA = std::min(unsigned(src_pos1), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos1 + 1), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos1 + width), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos1 + width + 1), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
										unsigned indexA = std::min(unsigned(src_pos2), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos2 + channels), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos2 + width * channels), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 图像绕任意点旋转某个角度，旋转后图像的宽度和高度不变
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">旋转变换后的tensor</param>
			/// <param name="center">指定的旋转中心</param>
			/// <param name="theta">旋转角度，逆时针方向为正</param>
			/// <param name="scale">图像缩放比例，默认为1，即保持不变</param>
			/// <param name="fill">空白区域填充值，默认为0</param>
			/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
			template <typename Dtype, typename Ptype>
			static void rotate_with_points_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst,
				const point<Ptype> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear)
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
				unsigned maxIndex = height * width * channels - 1;

				double rad = theta*(PI / 180);
				double cosa = cos(rad);
				double sina = sin(rad);

				double a = scale * cosa;
				double b = scale * sina;
				std::vector<std::vector<double> > M;
				std::vector<std::vector<double> > reverse_M;
				M.resize(3);
				for (size_t i = 0; i < M.size(); i++)
				{
					M[i].resize(3);
				}

				M[0][0] = a;
				M[0][1] = b;
				M[0][2] = (1 - a) * (double)center.x - b * (double)center.y;
				M[1][0] = -1 * b;
				M[1][1] = a;
				M[1][2] = b * (double)center.x + (1 - a) * (double)center.y;
				M[2][0] = 0;
				M[2][1] = 0;
				M[2][2] = 1;

				bool isInverted = math_functions::GetMatrixInverse(M, reverse_M);
				if (!isInverted)
				{
					LOG(FATAL) << "cannot rotate!!!";
					return;
				}

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
					const Dtype* src_data = src->cpu_data();
					Dtype* dst_data = dst->mutable_cpu_data();

					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

#pragma omp parallel for
						for (int row = 0; row < height; ++row)
						{
							double temp_xf = reverse_M[0][1] * row + reverse_M[0][2];
							double temp_yf = reverse_M[1][1] * row + reverse_M[1][2];
							int temp_dst_index = channel_offset + row * width;

							for (int col = 0; col < width; ++col)
							{
								double xf = reverse_M[0][0] * col + temp_xf;
								double yf = reverse_M[1][0] * col + temp_yf;
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
										unsigned indexA = std::min(unsigned(src_index), maxIndex);
										unsigned indexB = std::min(unsigned(src_index + 1), maxIndex);
										unsigned indexC = std::min(unsigned(src_index + width), maxIndex);
										unsigned indexD = std::min(unsigned(src_index + width + 1), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
					const Dtype* src_data = src->cpu_data();
					Dtype* dst_data = dst->mutable_cpu_data();

#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						double temp_xf = reverse_M[0][1] * row + reverse_M[0][2];
						double temp_yf = reverse_M[1][1] * row + reverse_M[1][2];
						int dst_pos1 = row * width * channels;

						for (int col = 0; col < width; ++col)
						{
							double xf = reverse_M[0][0] * col + temp_xf;
							double yf = reverse_M[1][0] * col + temp_yf;
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
										unsigned indexA = std::min(unsigned(src_pos2), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos2 + channels), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos2 + width * channels), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 图像绕任意点旋转某个角度，旋转后图像的宽度和高度不变
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">旋转变换后的tensor</param>
			/// <param name="center">指定的旋转中心</param>
			/// <param name="theta">旋转角度，逆时针方向为正</param>
			/// <param name="scale">图像缩放比例，默认为1，即保持不变</param>
			/// <param name="fill">空白区域填充值，默认为0</param>
			/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
			template <typename Dtype, typename Ptype>
			static void rotate_with_points_cpu(const tensor<Dtype> &src, tensor<Dtype> &dst,
				const point<Ptype> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear)
			{
				if (fabs(theta) <= 1e-6)
				{
					LOG(WARNING) << "Just copy from the source.";
					dst = src.clone();
					return;
				}

				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				unsigned maxIndex = height * width * channels - 1;

				double rad = theta*(PI / 180);
				double cosa = cos(rad);
				double sina = sin(rad);

				double a = scale * cosa;
				double b = scale * sina;

				std::vector<std::vector<double> > M;
				std::vector<std::vector<double> > reverse_M;
				M.resize(3);
				for (size_t i = 0; i < M.size(); i++)
				{
					M[i].resize(3);
				}

				M[0][0] = a;
				M[0][1] = b;
				M[0][2] = (1 - a) * (double)center.x - b * (double)center.y;
				M[1][0] = -1 * b;
				M[1][1] = a;
				M[1][2] = b * (double)center.x + (1 - a) * (double)center.y;
				M[2][0] = 0;
				M[2][1] = 0;
				M[2][2] = 1;

				bool isInverted = math_functions::GetMatrixInverse(M, reverse_M);
				if (!isInverted)
				{
					LOG(FATAL) << "cannot rotate!!!";
					return;
				}

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

#pragma omp parallel for
						for (int row = 0; row < height; ++row)
						{
							double temp_xf = reverse_M[0][1] * row + reverse_M[0][2];
							double temp_yf = reverse_M[1][1] * row + reverse_M[1][2];
							int temp_dst_index = channel_offset + row * width;

							for (int col = 0; col < width; ++col)
							{
								double xf = reverse_M[0][0] * col + temp_xf;
								double yf = reverse_M[1][0] * col + temp_yf;
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
										unsigned indexA = std::min(unsigned(src_index), maxIndex);
										unsigned indexB = std::min(unsigned(src_index + 1), maxIndex);
										unsigned indexC = std::min(unsigned(src_index + width), maxIndex);
										unsigned indexD = std::min(unsigned(src_index + width + 1), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						double temp_xf = reverse_M[0][1] * row + reverse_M[0][2];
						double temp_yf = reverse_M[1][1] * row + reverse_M[1][2];
						int dst_pos1 = row * width * channels;

						for (int col = 0; col < width; ++col)
						{
							double xf = reverse_M[0][0] * col + temp_xf;
							double yf = reverse_M[1][0] * col + temp_yf;
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
										unsigned indexA = std::min(unsigned(src_pos2), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos2 + channels), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos2 + width * channels), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 在图像上绘制矩形
			/// </summary>
			/// <param name="dst">包含图像数据的tensor</param>
			/// <param name="rect">待绘制的矩形</param>
			/// <param name="color_">线条颜色</param>
			/// <param name="thickness">线条宽度，默认为2像素</param>
			template <typename Dtype, typename Rtype>
			static void draw_rectangle_cpu(std::shared_ptr<tensor<Dtype>>& dst, rectangle<Rtype> rect, color color_, int thickness = 2)
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

					if (dst->order() == NCHW)
					{
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
					else if (dst->order() == NHWC)
					{
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
					else
					{
						NOT_IMPLEMENTED;
					}

					delete fill_color;
				}
				else
				{
					LOG(FATAL) << "Illegal channel numbers. Return without any changes.";
				}
			}



			/// <summary>
			/// 在图像上绘制矩形
			/// </summary>
			/// <param name="dst">包含图像数据的tensor</param>
			/// <param name="rect">待绘制的矩形</param>
			/// <param name="color_">线条颜色</param>
			/// <param name="thickness">线条宽度，默认为2像素</param>
			template <typename Dtype, typename Rtype>
			static void draw_rectangle_cpu(tensor<Dtype>& dst, rectangle<Rtype> rect, color color_, int thickness = 2)
			{
				CHECK_EQ(dst.num(), 1);
				int channels = dst.channels();
				int height = dst.height();
				int width = dst.width();
				int src_offset = height * width;
				Dtype* dst_data = dst.mutable_cpu_data();

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

					if (dst.order() == NCHW)
					{
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
					else if (dst.order() == NHWC)
					{
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
					else
					{
						NOT_IMPLEMENTED;
					}

					delete fill_color;
				}
				else
				{
					LOG(FATAL) << "Illegal channel numbers. Return without any changes.";
				}
			}



			/// <summary>
			/// 在图像上绘制圆
			/// </summary>
			/// <param name="dst">包含图像数据的tensor</param>
			/// <param name="center">圆心</param>
			/// <param name="radius">半径</param>
			/// <param name="color_">线条颜色</param>
			/// <param name="thickness">线条宽度，默认为2像素</param>
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

								if (dst->order() == NCHW)
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
								else if (dst->order() == NHWC)
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
								else
								{
									NOT_IMPLEMENTED;
								}

								delete fill_color;
							}
							else
							{
								LOG(FATAL) << "Illegal channel numbers. Return without any changes.";
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

							if (dst->order() == NCHW)
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
							else if (dst->order() == NHWC)
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
							else
							{
								NOT_IMPLEMENTED;
							}

							delete fill_color;
						}
						else
						{
							LOG(FATAL) << "Illegal channel numbers. Return without any changes.";
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



			/// <summary>
			/// 在图像上绘制圆
			/// </summary>
			/// <param name="dst">包含图像数据的tensor</param>
			/// <param name="center">圆心</param>
			/// <param name="radius">半径</param>
			/// <param name="color_">线条颜色</param>
			/// <param name="thickness">线条宽度，默认为2像素</param>
			template <typename Dtype, typename Rtype>
			static void draw_circle_cpu(tensor<Dtype>& dst, point<Rtype> center, int radius, color color_, int thickness = 2)
			{
				CHECK_EQ(dst.num(), 1);
				int channels = dst.channels();
				int height = dst.height();
				int width = dst.width();
				int offset = height * width;
				Dtype* dst_data = dst.mutable_cpu_data();

				if (center.x < radius || center.x >= width - radius && center.y < radius && center.y >= height - radius)
				{
					LOG(WARNING) << "circle out of image, return without any changes.";
					return;
				}

				if (thickness > 0)
				{
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

								if (dst.order() == NCHW)
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
								else if (dst.order() == NHWC)
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
								else
								{
									NOT_IMPLEMENTED;
								}

								delete fill_color;
							}
							else
							{
								LOG(FATAL) << "Illegal channel numbers. Return without any changes.";
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

							if (dst.order() == NCHW)
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
							else if (dst.order() == NHWC)
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
							else
							{
								NOT_IMPLEMENTED;
							}

							delete fill_color;
						}
						else
						{
							LOG(FATAL) << "Illegal channel numbers. Return without any changes.";
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



			/// <summary>
			/// 在图像上完成四种翻转：仅宽度方向翻转、仅高度方向翻转、宽度和高度同时翻转、按图像通道翻转
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">翻转变换后的tensor</param>
			/// <param name="axis">翻转轴：Width_Wise（仅宽度方向翻转，默认）、Height_Wise（仅高度方向翻转）、Center_Wise（宽度和高度同时翻转）、Channel_Wise（按图像通道翻转）</param>
			template <typename Dtype>
			static void flip_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, flipType axis = Width_Wise)
			{
				CHECK_EQ(src->num(), 1);
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 在图像上完成四种翻转：仅宽度方向翻转、仅高度方向翻转、宽度和高度同时翻转、按图像通道翻转
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">翻转变换后的tensor</param>
			/// <param name="axis">翻转轴：Width_Wise（仅宽度方向翻转，默认）、Height_Wise（仅高度方向翻转）、Center_Wise（宽度和高度同时翻转）、Channel_Wise（按图像通道翻转）</param>
			template <typename Dtype>
			static void flip_cpu(const tensor<Dtype> &src, tensor<Dtype>& dst, flipType axis = Width_Wise)
			{
				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 三通道rgb图像转换为单通道灰度图像
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">转换后的tensor</param>
			template <typename Dtype>
			static void rgb2gray_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
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

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->order()));
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->order()));
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 三通道rgb图像转换为单通道灰度图像
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">转换后的tensor</param>
			template <typename Dtype>
			static void rgb2gray_cpu(const tensor<Dtype> &src, tensor<Dtype>& dst)
			{
				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;

				if (channels != 3)
				{
					LOG(ERROR) << "Incorrect input channel.";
					return;
				}

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device(), src.type());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, 1}, src.device(), src.type());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 矩阵转置
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">转换后的tensor</param>
			template <typename Dtype>
			static void matrix_transpose_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
			{
				CHECK_EQ(src->num(), 1);
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, width, height}, src->device(), src->order()));
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, width, height, channels}, src->device(), src->order()));
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 矩阵转置
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">转换后的tensor</param>
			template <typename Dtype>
			static void matrix_transpose_cpu(const tensor<Dtype> &src, tensor<Dtype>& dst)
			{
				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, width, height}, src.device(), src.type());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, width, height, channels}, src.device(), src.type());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

					for (int row = 0; row < width; ++row)
					{
						int dst_pos1 = row * height * channels;
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 从图像中获取感兴趣区域
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">包含感兴趣区域图像数据的tensor</param>
			/// <param name="rect">矩形的感兴趣区域</param>
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

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 从图像中获取感兴趣区域
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">包含感兴趣区域图像数据的tensor</param>
			/// <param name="rect">矩形的感兴趣区域</param>
			template <typename Dtype, typename Rtype>
			static void roi_cpu(const tensor<Dtype> &src, tensor<Dtype>& dst, rectangle<Rtype> rect)
			{
				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;

				if (rect.x < 0 || rect.x + rect.w >= width || rect.y < 0 || rect.y + rect.h >= height)
				{
					LOG(WARNING) << "rect is out of image";
					return;
				}

				int dst_height = (int)rect.h;
				int dst_width = (int)rect.w;
				int dst_offset = dst_height * dst_width;

				if (dst_height == height && dst_width == width)
				{
					LOG(WARNING) << "Just copy from the source.";
					dst = src.clone();
					return;
				}

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 边界填充
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">填充后的tensor</param>
			/// <param name="top">图像上方的填充高度</param>
			/// <param name="bottom">图像下方的填充高度</param>
			/// <param name="left">图像左侧的填充宽度</param>
			/// <param name="right">图像右侧的填充宽度</param>
			/// <param name="type">边界类型：Border_Constant（用第8个参数fill设定的常量值进行填充，默认值）、Border_Replicate（复制相邻像素点的值进行填充）</param>
			/// <param name="fill">边界类型为Border_Constant时有效，默认为0</param>
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

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
					Dtype* dst_data = dst->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					if (type == Border_Constant)
					{
						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							//top
							for (int row = 0; row < top; row++)
							{
								int dst_index = dst_channel_offset + row * dst_width;
								for (int col = 0; col < dst_width; col++)
								{
									dst_data[dst_index + col] = (Dtype)fill;
								}
							}

							//center
							for (int row = top; row < top + height; ++row)
							{
								int src_index = src_channel_offset + (row - top) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								for (int col = 0; col < left; col++)
								{
									dst_data[dst_index + col] = (Dtype)fill;
								}

								memcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype));

								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_index + col] = (Dtype)fill;
								}
							}

							//bottom
							for (int row = top + height; row < dst_height; row++)
							{
								int dst_index = dst_channel_offset + row * dst_width;
								for (int col = 0; col < dst_width; col++)
								{
									dst_data[dst_index + col] = (Dtype)fill;
								}
							}
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

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_index + col] = src_data[src_channel_offset];
								}

								//center
								memcpy(dst_data + dst_index + left, src_data + src_channel_offset, width * sizeof(Dtype));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_index + col] = src_data[src_channel_offset + width - 1];
								}
							}

							//center
							for (int row = top; row < top + height; ++row)
							{
								int src_index = src_channel_offset + (row - top) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_index + col] = src_data[src_index];
								}

								//center
								memcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_index + col] = src_data[src_index + width - 1];
								}
							}

							//bottom
							for (int row = top + height; row < dst_height; ++row)
							{
								int src_index = src_channel_offset + (height - 1) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_index + col] = src_data[src_index];
								}

								//center
								memcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_index + col] = src_data[src_index + width - 1];
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					if (type == Border_Constant)
					{
						//top
						for (int row = 0; row < top; row++)
						{
							int dst_index1 = row * dst_width * channels;
							for (int col = 0; col < dst_width; col++)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ch++)
								{
									dst_data[dst_index2 + ch] = (Dtype)fill;
								}
							}
						}

						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index = (row - top) * width * channels;

							//left
							int dst_index1 = row * dst_width * channels;
							for (int col = 0; col < left; col++)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ch++)
								{
									dst_data[dst_index2 + ch] = (Dtype)fill;
								}
							}

							//center
							memcpy(dst_data + dst_index1 + left * channels, src_data + src_index, width * channels * sizeof(Dtype));

							//right
							for (int col = left + width; col < dst_width; col++)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ch++)
								{
									dst_data[dst_index2 + ch] = (Dtype)fill;
								}
							}
						}

						//bottom
						for (int row = top + height; row < dst_height; row++)
						{
							int dst_index1 = row * dst_width * channels;
							for (int col = 0; col < dst_width; col++)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ch++)
								{
									dst_data[dst_index2 + ch] = (Dtype)fill;
								}
							}
						}
					}
					else if (type == Border_Replicate)
					{
						//top
						for (int row = 0; row < top; ++row)
						{
							int dst_index1 = row * dst_width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_index2 + ch] = src_data[ch];
								}
							}

							//center
							memcpy(dst_data + dst_index1 + left * channels, src_data, width * channels * sizeof(Dtype));

							//right
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								int src_index = (width - 1) * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_index2 + ch] = src_data[src_index + ch];
								}
							}
						}


						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index1 = (row - top) * width * channels;
							int dst_index1 = row * dst_width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_index2 + ch] = src_data[src_index1 + ch];
								}
							}

							//center
							memcpy(dst_data + dst_index1 + left * channels, src_data + src_index1, width * channels * sizeof(Dtype));

							//right
							int src_index2 = src_index1 + (width - 1) * channels;
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_index2 + ch] = src_data[src_index2 + ch];
								}
							}
						}


						//bottom
						for (int row = top + height; row < dst_height; ++row)
						{
							int dst_index1 = row * dst_width * channels;
							int src_index1 = (height - 1) * width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_index2 + ch] = src_data[src_index1 + ch];
								}
							}

							//center
							memcpy(dst_data + dst_index1 + left * channels, src_data + src_index1, width * channels * sizeof(Dtype));

							//right
							int src_index2 = src_index1 + (width - 1) * channels;
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_index2 + ch] = src_data[src_index2 + ch];
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 边界填充
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">填充后的tensor</param>
			/// <param name="top">图像上方的填充高度</param>
			/// <param name="bottom">图像下方的填充高度</param>
			/// <param name="left">图像左侧的填充宽度</param>
			/// <param name="right">图像右侧的填充宽度</param>
			/// <param name="type">边界类型：Border_Constant（用第8个参数fill设定的常量值进行填充，默认值）、Border_Replicate（复制相邻像素点的值进行填充）</param>
			/// <param name="fill">边界类型为Border_Constant时有效，默认为0</param>
			template <typename Dtype>
			static void make_border_cpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
				int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0)
			{
				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
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
					dst = src.clone();
					return;
				}

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 边界裁剪
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">裁剪后的tensor</param>
			/// <param name="top">图像上方的裁剪高度</param>
			/// <param name="bottom">图像下方的裁剪高度</param>
			/// <param name="left">图像左侧的裁剪宽度</param>
			/// <param name="right">图像右侧的裁剪宽度</param>
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

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					for (int row = 0; row < dst_height; ++row)
					{
						int src_index = ((row + top) * width + left) * channels;
						int dst_index = row * dst_width * channels;
						memcpy(dst_data + dst_index, src_data + src_index, dst_width * channels * sizeof(Dtype));
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 边界裁剪
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">裁剪后的tensor</param>
			/// <param name="top">图像上方的裁剪高度</param>
			/// <param name="bottom">图像下方的裁剪高度</param>
			/// <param name="left">图像左侧的裁剪宽度</param>
			/// <param name="right">图像右侧的裁剪宽度</param>
			template <typename Dtype>
			static void cut_border_cpu(const tensor<Dtype> &src,
				tensor<Dtype>& dst, int top, int bottom, int left, int right)
			{
				if (top < 0 || bottom < 0 || left < 0 || right < 0)
				{
					LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
					return;
				}

				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;

				int dst_height = height - top - bottom;
				int dst_width = width - left - right;
				int dst_offset = dst_height * dst_width;

				if (dst_height == height && dst_width == width)
				{
					LOG(WARNING) << "Just copy from the source.";
					dst = src.clone();
					return;
				}

				if (dst_height <= 0 || dst_width <= 0)
				{
					LOG(ERROR) << "Illegal input size.";
					return;
				}

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					for (int row = 0; row < dst_height; ++row)
					{
						int src_index = ((row + top) * width + left) * channels;
						int dst_index = row * dst_width * channels;
						memcpy(dst_data + dst_index, src_data + src_index, dst_width * channels * sizeof(Dtype));
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 对w×h的图像，计算(w+1)×(h+1)的积分图，其中第一行、第一列值全为0
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">包含计算后的积分图像数据的tensor</param>
			template <typename Stype, typename Dtype>
			static void fast_integral_cpu(const std::shared_ptr<tensor<Stype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
			{
				CHECK_EQ(src->num(), 1);
				int channels = src->channels();
				int height = src->height();
				int width = src->width();

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height + 1, width + 1}, src->device(), src->order()));
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height + 1, width + 1, channels}, src->device(), src->order()));
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 对w×h的图像，计算(w+1)×(h+1)的积分图，其中第一行、第一列值全为0
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">包含计算后的积分图像数据的tensor</param>
			template <typename Stype, typename Dtype>
			static void fast_integral_cpu(const tensor<Stype> &src, tensor<Dtype>& dst)
			{
				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, height + 1, width + 1}, src.device(), src.type());
					const Stype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();
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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height + 1, width + 1, channels}, src.device(), src.type());
					const Stype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 直方图均衡化
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">直方图均衡化后的tensor</param>
			template <typename Dtype>
			static void equalize_hist_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
			{
				CHECK_EQ(src->num(), 1);
				CHECK_EQ(src->channels(), 1);
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				if (dst->count() != src->count())
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device()));
				}
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



			/// <summary>
			/// 直方图均衡化
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">直方图均衡化后的tensor</param>
			template <typename Dtype>
			static void equalize_hist_cpu(const tensor<Dtype> &src, tensor<Dtype>& dst)
			{
				CHECK_EQ(src.num(), 1);
				CHECK_EQ(src.channels(), 1);
				int height = src.height();
				int width = src.width();
				int offset = height * width;

				if (dst.count() != src.count())
				{
					dst = tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device());
				}
				
				const Dtype* src_data = src.cpu_data();
				Dtype* dst_data = dst.mutable_cpu_data();

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



			/// <summary>
			/// 将一张多通道图像分离为多张单通道图像
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst_vector">一组tensor，每个tensor中包含了原图一个通道的数据</param>
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

				if (src->order() == NCHW)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * src_offset;
						std::shared_ptr<tensor<Dtype>> temp_ptr;
						temp_ptr.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->order()));
						Dtype* temp_data = temp_ptr->mutable_cpu_data();
						std::memcpy((void*)temp_data, (void*)(src_data + channel_offset), src_offset * sizeof(Dtype));
						dst_vector.push_back(temp_ptr);
					}
				}
				else if (src->order() == NHWC)
				{
					dst_vector.resize(channels);
					std::vector<Dtype *> ptr_arr(channels);

					for (int ch = 0; ch < channels; ++ch)
					{
						dst_vector.at(ch).reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->order()));
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 将一张多通道图像分离为多张单通道图像
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst_vector">一组tensor，每个tensor中包含了原图一个通道的数据</param>
			template <typename Dtype>
			static void split_channel_cpu(const tensor<Dtype> &src, std::vector<tensor<Dtype>>& dst_vector)
			{
				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;

				const Dtype* src_data = src.cpu_data();
				dst_vector.clear();

				if (src.order() == NCHW)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * src_offset;
						tensor<Dtype> temp_ptr = tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device(), src.type());
						Dtype* temp_data = temp_ptr.mutable_cpu_data();
						std::memcpy((void*)temp_data, (void*)(src_data + channel_offset), src_offset * sizeof(Dtype));
						dst_vector.push_back(temp_ptr);
					}
				}
				else if (src.order() == NHWC)
				{
					dst_vector.resize(channels);
					std::vector<Dtype *> ptr_arr(channels);

					for (int ch = 0; ch < channels; ++ch)
					{
						dst_vector.at(ch) = tensor<Dtype>(std::vector<int>{1, height, width, 1}, src.device(), src.type());
						ptr_arr[ch] = dst_vector.at(ch).mutable_cpu_data();
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 将多个tensor图像数据按vector索引下标顺序堆叠在一起，各tensor图像之间仅需保证宽度、高度、GPU/CPU、排列方式相同；每个tensor可以有不同的图像数量和通道数量
			/// </summary>
			/// <param name="src_vector">一组tensor，每个tensor中包含了宽度、高度、GPU/CPU、排列方式相同的图像数据</param>
			/// <param name="dst">堆叠后产生的tensor，按照vector索引下标顺序堆叠</param>
			template <typename Dtype>
			static void merge_channel_cpu(const std::vector<std::shared_ptr<tensor<Dtype>>> &src_vector, std::shared_ptr<tensor<Dtype>> &dst)
			{
				CHECK_EQ(src_vector.size(), 3);
				int height, width, device;
				orderType type;
				for (int i = 0; i < src_vector.size(); ++i)
				{
					CHECK_EQ(src_vector.at(i)->channels(), 1);
					if (i == 0)
					{
						height = src_vector.at(i)->height();
						width = src_vector.at(i)->width();
						device = src_vector.at(i)->device();
						type = src_vector.at(i)->order();
					}
					else
					{
						if (height != src_vector.at(i)->height() ||
							width != src_vector.at(i)->width() ||
							device != src_vector.at(i)->device() ||
							type != src_vector.at(i)->order())
						{
							LOG(WARNING) << "the element of vector<mat> should have the exact same height/width/device/type.";
							return;
						}
					}
				}

				if (type == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, 3, height, width}, device, type));
				}
				else if (type == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 3}, device, type));
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				Dtype* dst_data = dst->mutable_cpu_data();
				int offset = height * width;

				for (int i = 0; i < src_vector.size(); ++i)
				{
					const Dtype* temp_data = src_vector.at(i)->cpu_data();
					std::memcpy((void*)(dst_data + i * offset), (void*)(temp_data), offset * sizeof(Dtype));
				}
			}



			/// <summary>
			/// 将多个tensor图像数据按vector索引下标顺序堆叠在一起，各tensor图像之间仅需保证宽度、高度、GPU/CPU、排列方式相同；每个tensor可以有不同的图像数量和通道数量
			/// </summary>
			/// <param name="src_vector">一组tensor，每个tensor中包含了宽度、高度、GPU/CPU、排列方式相同的图像数据</param>
			/// <param name="dst">堆叠后产生的tensor，按照vector索引下标顺序堆叠</param>
			template <typename Dtype>
			static void merge_channel_cpu(const std::vector<tensor<Dtype>> &src_vector, tensor<Dtype> &dst)
			{
				CHECK_EQ(src_vector.size(), 3);
				int height, width, device;
				orderType type;
				for (int i = 0; i < src_vector.size(); ++i)
				{
					CHECK_EQ(src_vector.at(i)->channels(), 1);
					if (i == 0)
					{
						height = src_vector.at(i).height();
						width = src_vector.at(i).width();
						device = src_vector.at(i).device();
						type = src_vector.at(i).type();
					}
					else
					{
						if (height != src_vector.at(i).height() ||
							width != src_vector.at(i).width() ||
							device != src_vector.at(i).device() ||
							type != src_vector.at(i).type())
						{
							LOG(WARNING) << "the element of vector<mat> should have the exact same height/width/device/type.";
							return;
						}
					}
				}

				if (type == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, 3, height, width}, device, type);
				}
				else if (type == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, 3}, device, type);
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				Dtype* dst_data = dst.mutable_cpu_data();
				int offset = height * width;

				for (int i = 0; i < src_vector.size(); ++i)
				{
					const Dtype* temp_data = src_vector.at(i).cpu_data();
					std::memcpy((void*)(dst_data + i * offset), (void*)(temp_data), offset * sizeof(Dtype));
				}
			}



			/// <summary>
			/// 二值化灰度图像
			/// </summary>
			/// <param name="src">包含原图像数据的tensor，单通道灰度图</param>
			/// <param name="dst">二值化后的tensor</param>
			/// <param name="thresh">阈值，默认为128</param>
			/// <param name="maxval">最大值，默认为255</param>
			/// <param name="type">二值化类型：binary（灰度值大于thresh的像素点，将灰度值设为maxval，反之设为0，默认值）、
			///                           binary_inv（灰度值小于thresh的像素点，将灰度值设为maxval，反之设为0）、
			///                          small_trunc（灰度值小于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
			///                            big_trunc（灰度值大于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
			///                        small_to_zero（灰度值小于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）、
			///                          big_to_zero（灰度值大于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）</param>
			template <typename Dtype>
			static void threshold_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary)
			{
				CHECK_EQ(src->num(), 1);
				CHECK_EQ(src->channels(), 1);
				int height = src->height();
				int width = src->width();
				int offset = height * width;

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->order()));
				}
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->order()));
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				Dtype* dst_data = dst->mutable_cpu_data();
				const Dtype *src_data = src->cpu_data();

				switch (type)
				{
				case binary:
					for (int j = 0; j < offset; ++j)
					{
						dst_data[j] = src_data[j] >(Dtype)thresh ? (Dtype)maxval : 0;
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



			/// <summary>
			/// 二值化灰度图像
			/// </summary>
			/// <param name="src">包含原图像数据的tensor，单通道灰度图</param>
			/// <param name="dst">二值化后的tensor</param>
			/// <param name="thresh">阈值，默认为128</param>
			/// <param name="maxval">最大值，默认为255</param>
			/// <param name="type">二值化类型：binary（灰度值大于thresh的像素点，将灰度值设为maxval，反之设为0，默认值）、
			///                           binary_inv（灰度值小于thresh的像素点，将灰度值设为maxval，反之设为0）、
			///                          small_trunc（灰度值小于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
			///                            big_trunc（灰度值大于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
			///                        small_to_zero（灰度值小于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）、
			///                          big_to_zero（灰度值大于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）</param>
			template <typename Dtype>
			static void threshold_cpu(const tensor<Dtype> &src, tensor<Dtype>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary)
			{
				CHECK_EQ(src.num(), 1);
				CHECK_EQ(src.channels(), 1);
				int height = src.height();
				int width = src.width();
				int offset = height * width;

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device(), src.type());
				}
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, 1}, src.device(), src.type());
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				Dtype* dst_data = dst.mutable_cpu_data();
				const Dtype *src_data = src.cpu_data();

				switch (type)
				{
				case binary:
					for (int j = 0; j < offset; ++j)
					{
						dst_data[j] = src_data[j] >(Dtype)thresh ? (Dtype)maxval : 0;
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






			/// <summary>
			/// 仿射变换
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">变换后的tensor</param>
			/// <param name="src_point">变换前的3个点</param>
			/// <param name="dst_point">变换后对应的3个点，可通过src_point和dst_point计算得到变换矩阵，进而使用该变换矩阵对整张图像完成仿射变换</param>
			/// <param name="fill">空白区域的填充值，默认为0</param>
			/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
			template <typename Dtype, typename Ptype>
			static void warp_affine_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst,
				const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill = 0, interpolationType type = Bilinear)
			{
				if (src_point.size() != 3 || dst_point.size() != 3)
				{
					LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
					return;
				}

				CHECK_EQ(src->num(), 1);
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				unsigned maxIndex = height * width * channels - 1;

				//AX=B，先扩充至6×6和6×1
				std::vector<std::vector<float> > A;
				std::vector<float> B, X;
				A.resize(6);
				for (size_t i = 0; i < 6; i = i + 2)
				{
					A[i].push_back(float(dst_point[i / 2].x));
					A[i].push_back(float(dst_point[i / 2].y));
					A[i].push_back(1.0f);
					A[i].push_back(0.0f);
					A[i].push_back(0.0f);
					A[i].push_back(0.0f);

					A[i + 1].push_back(0.0f);
					A[i + 1].push_back(0.0f);
					A[i + 1].push_back(0.0f);
					A[i + 1].push_back(float(dst_point[i / 2].x));
					A[i + 1].push_back(float(dst_point[i / 2].y));
					A[i + 1].push_back(1.0f);
				}

				for (size_t i = 0; i < 3; i++)
				{
					B.push_back(float(src_point[i].x));
					B.push_back(float(src_point[i].y));
				}

				X = math_functions::gauss_all(A, B);

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
					const Dtype* src_data = src->cpu_data();
					Dtype* dst_data = dst->mutable_cpu_data();

					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

			#pragma omp parallel for
						for (int row = 0; row < height; ++row)
						{
							double temp_xf = X[1] * row + X[2];
							double temp_yf = X[4] * row + X[5];
							int temp_dst_index = channel_offset + row * width;

							for (int col = 0; col < width; ++col)
							{
								double xf = X[0] * col + temp_xf;
								double yf = X[3] * col + temp_yf;
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
										unsigned indexA = std::min(unsigned(src_index), maxIndex);
										unsigned indexB = std::min(unsigned(src_index + 1), maxIndex);
										unsigned indexC = std::min(unsigned(src_index + width), maxIndex);
										unsigned indexD = std::min(unsigned(src_index + width + 1), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
					const Dtype* src_data = src->cpu_data();
					Dtype* dst_data = dst->mutable_cpu_data();

			#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						double temp_xf = X[1] * row + X[2];
						double temp_yf = X[4] * row + X[5];
						int dst_pos1 = row * width * channels;

						for (int col = 0; col < width; ++col)
						{
							double xf = X[0] * col + temp_xf;
							double yf = X[3] * col + temp_yf;
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
										unsigned indexA = std::min(unsigned(src_pos1 + ch), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos1 + channels + ch), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos1 + width * channels + ch), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos1 + (width + 1) * channels + ch), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else
				{
					NOT_IMPLEMENTED;
				}

			}



			/// <summary>
			/// 仿射变换
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">变换后的tensor</param>
			/// <param name="src_point">变换前的3个点</param>
			/// <param name="dst_point">变换后对应的3个点，可通过src_point和dst_point计算得到变换矩阵，进而使用该变换矩阵对整张图像完成仿射变换</param>
			/// <param name="fill">空白区域的填充值，默认为0</param>
			/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
			template <typename Dtype, typename Ptype>
			static void warp_affine_cpu(const tensor<Dtype> &src, tensor<Dtype> &dst,
				const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill = 0, interpolationType type = Bilinear)
			{
				if (src_point.size() != 3 || dst_point.size() != 3)
				{
					LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
					return;
				}

				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				unsigned maxIndex = height * width * channels - 1;

				//AX=B，先扩充至6×6和6×1
				std::vector<std::vector<float> > A;
				std::vector<float> B, X;
				A.resize(6);
				for (size_t i = 0; i < 6; i = i + 2)
				{
					A[i].push_back(float(dst_point[i / 2].x));
					A[i].push_back(float(dst_point[i / 2].y));
					A[i].push_back(1.0f);
					A[i].push_back(0.0f);
					A[i].push_back(0.0f);
					A[i].push_back(0.0f);

					A[i + 1].push_back(0.0f);
					A[i + 1].push_back(0.0f);
					A[i + 1].push_back(0.0f);
					A[i + 1].push_back(float(dst_point[i / 2].x));
					A[i + 1].push_back(float(dst_point[i / 2].y));
					A[i + 1].push_back(1.0f);
				}

				for (size_t i = 0; i < 3; i++)
				{
					B.push_back(float(src_point[i].x));
					B.push_back(float(src_point[i].y));
				}

				X = math_functions::gauss_all(A, B);

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

					for (int ch = 0; ch < channels; ++ch)
					{
						int channel_offset = ch * offset;

#pragma omp parallel for
						for (int row = 0; row < height; ++row)
						{
							double temp_xf = X[1] * row + X[2];
							double temp_yf = X[4] * row + X[5];
							int temp_dst_index = channel_offset + row * width;

							for (int col = 0; col < width; ++col)
							{
								double xf = X[0] * col + temp_xf;
								double yf = X[3] * col + temp_yf;
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
										unsigned indexA = std::min(unsigned(src_index), maxIndex);
										unsigned indexB = std::min(unsigned(src_index + 1), maxIndex);
										unsigned indexC = std::min(unsigned(src_index + width), maxIndex);
										unsigned indexD = std::min(unsigned(src_index + width + 1), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();

#pragma omp parallel for
					for (int row = 0; row < height; ++row)
					{
						double temp_xf = X[1] * row + X[2];
						double temp_yf = X[4] * row + X[5];
						int dst_pos1 = row * width * channels;

						for (int col = 0; col < width; ++col)
						{
							double xf = X[0] * col + temp_xf;
							double yf = X[3] * col + temp_yf;
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
										unsigned indexA = std::min(unsigned(src_pos1 + ch), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos1 + channels + ch), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos1 + width * channels + ch), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos1 + (width + 1) * channels + ch), maxIndex);
										Dtype A = src_data[indexA];
										Dtype B = src_data[indexB];
										Dtype C = src_data[indexC];
										Dtype D = src_data[indexD];

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 高斯滤波
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">高斯滤波后的tensor</param>
			/// <param name="ksize">高斯核的尺寸，只能为奇数，默认为3。实际使用的高斯模板规格为ksize×ksize的矩形窗</param>
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

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
					Dtype* dst_data = dst->mutable_cpu_data();

					std::shared_ptr<tensor<Dtype>> temp;
					temp.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst->mutable_cpu_data();

					std::shared_ptr<tensor<Dtype>> temp;
					temp.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 高斯滤波
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">高斯滤波后的tensor</param>
			/// <param name="ksize">高斯核的尺寸，只能为奇数，默认为3。实际使用的高斯模板规格为ksize×ksize的矩形窗</param>
			template <typename Dtype>
			static void gaussian_blur_cpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int ksize = 3)
			{
				if (ksize % 2 != 1)
				{
					LOG(WARNING) << "convolution kernel: width and height should be odd.";
					return;
				}

				if (ksize == 1)
				{
					LOG(WARNING) << "Just copy from the source.";
					dst = src.clone();
					return;
				}

				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
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

				const Dtype* src_data = src.cpu_data();

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();

					tensor<Dtype> temp = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
					Dtype* temp_data = temp.mutable_cpu_data();

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();

					tensor<Dtype> temp = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
					Dtype* temp_data = temp.mutable_cpu_data();

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 计算水平、竖直方向的梯度边缘
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">计算后的tensor</param>
			/// <param name="dx">dx>0时，沿着x方向求导，得到竖直边缘；dx小于等于0时，不计算竖直边缘。默认为1</param>
			/// <param name="dy">dy>0时，沿着y方向求导，得到水平边缘；dy小于等于0时，不计算水平边缘。默认为1</param>
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

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
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
									total = 0.25 * total;
								}
								else
								{
									total = 0.5 * total;
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
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
									total = 0.25 * total;
								}
								else
								{
									total = 0.5 * total;
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 计算水平、竖直方向的梯度边缘
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">计算后的tensor</param>
			/// <param name="dx">dx>0时，沿着x方向求导，得到竖直边缘；dx小于等于0时，不计算竖直边缘。默认为1</param>
			/// <param name="dy">dy>0时，沿着y方向求导，得到水平边缘；dy小于等于0时，不计算水平边缘。默认为1</param>
			template <typename Dtype>
			static void sobel_cpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int dx = 1, int dy = 1)
			{
				if (dx == 0 && dy == 0)
				{
					LOG(WARNING) << "dx, dy cannot be zero simultaneously.";
					return;
				}

				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
									total = 0.25 * total;
								}
								else
								{
									total = 0.5 * total;
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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
									total = 0.25 * total;
								}
								else
								{
									total = 0.5 * total;
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 对图像进行膨胀、腐蚀处理
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">膨胀、腐蚀后的tensor</param>
			/// <param name="type">形态学类型：Dilate（膨胀，默认值）、Erode（腐蚀）</param>
			/// <param name="ksize">矩形核的尺寸，只能为奇数，默认为3。实际使用的矩形规格为ksize×ksize</param>
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

				if (src->order() == NCHW)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
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
				else if (src->order() == NHWC)
				{
					dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// 对图像进行膨胀、腐蚀处理
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">膨胀、腐蚀后的tensor</param>
			/// <param name="type">形态学类型：Dilate（膨胀，默认值）、Erode（腐蚀）</param>
			/// <param name="ksize">矩形核的尺寸，只能为奇数，默认为3。实际使用的矩形规格为ksize×ksize</param>
			template <typename Dtype>
			static void morph_cpu(const tensor<Dtype> &src, tensor<Dtype> &dst, morphType type = Dilate, int ksize = 3)
			{
				if (ksize % 2 != 1)
				{
					LOG(WARNING) << "ksize should be odd.";
					return;
				}

				if (ksize == 1)
				{
					dst = src.clone();
					LOG(WARNING) << "Just copy from the source.";
					return;
				}

				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int half = (ksize - 1) * 0.5;

				if (src.order() == NCHW)
				{
					dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
				else if (src.order() == NHWC)
				{
					dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

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
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			template <typename Dtype>
			/// <summary>
			/// 对w×h的图像，计算（w-2)×(h-2)的LBP特征
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">包含LBP特征值的新tensor</param>
			/// <param name="type">类型：Native（对每个像素点，在3×3矩形邻域找到相邻8个点进行计算）</param>
			static void lbp_feature_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, lbpType type = Native)
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
					if (src->order() == NCHW)
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
					else if (src->order() == NHWC)
					{
						for (size_t h = 1; h < height - 1; h++)
						{
							int offset = h * width * channels;
							int offset_plus = offset + width * channels;
							int offset_minus = offset - width * channels;
							for (size_t w = 1; w < width - 1; w++)
							{
								for (size_t c = 0; c < channels; c++)
								{
									Dtype center = src_data[offset + w * channels + c];
									unsigned char code = 0;
									code |= (src_data[offset_minus + (w - 1) * channels + c] >= center) << 7;
									code |= (src_data[offset_minus + (w - 0) * channels + c] >= center) << 6;
									code |= (src_data[offset_minus + (w + 1) * channels + c] >= center) << 5;
									code |= (src_data[offset + (w + 1) * channels + c] >= center) << 4;
									code |= (src_data[offset_plus + (w + 1) * channels + c] >= center) << 3;
									code |= (src_data[offset_plus + (w + 0) * channels + c] >= center) << 2;
									code |= (src_data[offset_plus + (w - 1) * channels + c] >= center) << 1;
									code |= (src_data[offset + (w - 1) * channels + c] >= center) << 0;
									dst_data[dst_c_offset + ((h - 1) * (width - 2) + (w - 1)) * channels + c] = static_cast<Dtype>(code);
								}
							}
						}
					}
					else
					{
						NOT_IMPLEMENTED;
					}

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
			/// <summary>
			/// 对w×h的图像，计算（w-2)×(h-2)的LBP特征
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">包含LBP特征值的新tensor</param>
			/// <param name="type">类型：Native（对每个像素点，在3×3矩形邻域找到相邻8个点进行计算）</param>
			static void lbp_feature_cpu(const tensor<Dtype>& src, tensor<Dtype>& dst, lbpType type = Native)
			{
				CHECK_EQ(src.num(), 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				const Dtype* src_data = src.cpu_data();
				Dtype* dst_data = dst.mutable_cpu_data();
				switch (type)
				{
				case excalibur::Native:
					if (src->order() == NCHW)
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
					else if (src->order() == NHWC)
					{
						for (size_t h = 1; h < height - 1; h++)
						{
							int offset = h * width * channels;
							int offset_plus = offset + width * channels;
							int offset_minus = offset - width * channels;
							for (size_t w = 1; w < width - 1; w++)
							{
								for (size_t c = 0; c < channels; c++)
								{
									Dtype center = src_data[offset + w * channels + c];
									unsigned char code = 0;
									code |= (src_data[offset_minus + (w - 1) * channels + c] >= center) << 7;
									code |= (src_data[offset_minus + (w - 0) * channels + c] >= center) << 6;
									code |= (src_data[offset_minus + (w + 1) * channels + c] >= center) << 5;
									code |= (src_data[offset + (w + 1) * channels + c] >= center) << 4;
									code |= (src_data[offset_plus + (w + 1) * channels + c] >= center) << 3;
									code |= (src_data[offset_plus + (w + 0) * channels + c] >= center) << 2;
									code |= (src_data[offset_plus + (w - 1) * channels + c] >= center) << 1;
									code |= (src_data[offset + (w - 1) * channels + c] >= center) << 0;
									dst_data[dst_c_offset + ((h - 1) * (width - 2) + (w - 1)) * channels + c] = static_cast<Dtype>(code);
								}
							}
						}
					}
					else
					{
						NOT_IMPLEMENTED;
					}

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
			/// <summary>
			/// 分块计算LBP特征值
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">包含LBP特征值的新tensor</param>
			/// <param name="block_h">块的高度</param>
			/// <param name="block_w">块的宽度</param>
			/// <param name="stride_h">块在高度方向上的步进距离</param>
			/// <param name="stride_w">块在宽度方向上的步进距离</param>
			static void mblbp_feature_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int block_h,
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

				std::shared_ptr<tensor<Dtype>> src_tensor = std::make_shared<tensor<Dtype>>(src);
				std::shared_ptr<tensor<Dtype>> integral_tensor = std::make_shared<tensor<Dtype>>(tensor<Dtype>::tensor(src->order()));
				fast_integral_cpu(src_tensor, integral_tensor);
				Dtype* integral_data = integral_tensor->mutable_cpu_data();

				int dst_height = dst->height();
				int dst_width = dst->width();

				if (src->order() == NCHW)
				{
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
				}
				else if (src->order() == NHWC)
				{
					for (size_t h = 0; h < height - 3 * block_h + 1; h += stride_h)
					{
						int dst_offset = h / stride_h * dst_width * channels;
						for (size_t w = 0; w < width - 3 * block_w + 1; w += stride_w)
						{
							for (size_t c = 0; c < channels; c++)
							{
								Dtype block_values[9];
								for (size_t i = 0; i < 9; i++)
								{
									int x1 = w + (i % 3) * block_w;
									int y1 = h + (i / 3) * block_h;
									int x2 = x1 + block_w;
									int y2 = y1 + block_h;
									float A = (float)integral_data[(y1 * (height + 1) + x1) * channels + c];
									float B = (float)integral_data[(y1 * (height + 1) + x2) * channels + c];
									float C = (float)integral_data[(y2 * (height + 1) + x1) * channels + c];
									float D = (float)integral_data[(y2 * (height + 1) + x2) * channels + c];
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
								dst_data[dst_offset + w / stride_w * channels + c] = Dtype(code);
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			template <typename Dtype>
			/// <summary>
			/// 分块计算LBP特征值
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">包含LBP特征值的新tensor</param>
			/// <param name="block_h">块的高度</param>
			/// <param name="block_w">块的宽度</param>
			/// <param name="stride_h">块在高度方向上的步进距离</param>
			/// <param name="stride_w">块在宽度方向上的步进距离</param>
			static void mblbp_feature_cpu(const tensor<Dtype>& src, tensor<Dtype>& dst, int block_h,
				int block_w, int stride_h, int stride_w)
			{
				CHECK_EQ(src.num(), 1);
				CHECK_GE(block_h, 1);
				CHECK_GE(block_w, 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				CHECK_GE(height, 3 * block_h);
				CHECK_GE(width, 3 * block_w);
				const Dtype* src_data = src.cpu_data();
				Dtype* dst_data = dst.mutable_cpu_data();

				tensor<Dtype> integral_tensor = tensor<Dtype>::tensor(src->order());
				fast_integral_cpu(src, integral_tensor);
				Dtype* integral_data = integral_tensor.mutable_cpu_data();

				int dst_height = dst.height();
				int dst_width = dst.width();

				if (src.order() == NCHW)
				{
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
				}
				else if (src.order() == NHWC)
				{
					for (size_t h = 0; h < height - 3 * block_h + 1; h += stride_h)
					{
						int dst_offset = h / stride_h * dst_width * channels;
						for (size_t w = 0; w < width - 3 * block_w + 1; w += stride_w)
						{
							for (size_t c = 0; c < channels; c++)
							{
								Dtype block_values[9];
								for (size_t i = 0; i < 9; i++)
								{
									int x1 = w + (i % 3) * block_w;
									int y1 = h + (i / 3) * block_h;
									int x2 = x1 + block_w;
									int y2 = y1 + block_h;
									float A = (float)integral_data[(y1 * (height + 1) + x1) * channels + c];
									float B = (float)integral_data[(y1 * (height + 1) + x2) * channels + c];
									float C = (float)integral_data[(y2 * (height + 1) + x1) * channels + c];
									float D = (float)integral_data[(y2 * (height + 1) + x2) * channels + c];
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
								dst_data[dst_offset + w / stride_w * channels + c] = Dtype(code);
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// tensor预处理，按通道对每个像素点进行相同的乘法和加法变换
			/// </summary>
			/// <param name="dst">包含图像数据的tensor</param>
			template <typename Dtype>
			static void preprocess_tensors_cpu(std::shared_ptr<tensor<Dtype>> dst)
			{
				int num = dst->num();
				int channel = dst->channels();
				int height = dst->height();
				int width = dst->width();
				if (channel == 3)
				{
					Dtype* dst_data = dst->mutable_cpu_data();
					float means[] = { 104.f, 117.0f, 124.f };
					float var = 0.0078125f;

					if (dst->order() == NCHW)
					{
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
					else if (dst->order() == NHWC)
					{
						for (size_t n = 0; n < num; n++)
						{
							int offset = n * 3 * height * width;
							for (size_t h = 0; h < height; h++)
							{
								int sub_offset = h * width * 3;
								for (size_t w = 0; w < width; w++)
								{
									int subsub_offset = w * 3;
									for (size_t c = 0; c < 3; c++)
									{
										dst_data[offset + sub_offset + subsub_offset + c] =
											Dtype((dst_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
									}
								}
							}
						}
					}
					else
					{
						NOT_IMPLEMENTED;
					}
				}
				else if (channel == 1)
				{
					Dtype* dst_data = dst->mutable_cpu_data();
					float means[] = { 127.5f };
					float var = 0.0078125f;

					for (size_t n = 0; n < num; n++)
					{
						int offset = n * height * width;
						for (size_t h = 0; h < height; h++)
						{
							int subsub_offset = h * width;
							for (size_t w = 0; w < width; w++)
							{
								dst_data[offset + subsub_offset + w] =
									Dtype((dst_data[offset + subsub_offset + w] - means[0]) * var);
							}
						}
					}
				}
			}



			/// <summary>
			/// tensor预处理，按通道对每个像素点进行相同的乘法和加法变换
			/// </summary>
			/// <param name="dst">包含图像数据的tensor</param>
			template <typename Dtype>
			static void preprocess_tensors_cpu(tensor<Dtype>* dst)
			{
				int num = dst->num();
				int channel = dst->channels();
				int height = dst->height();
				int width = dst->width();

				if (channel == 3)
				{
					Dtype* dst_data = dst->mutable_cpu_data();
					float means[] = { 104.f, 117.0f, 124.f };
					float var = 0.0078125f;

					if (dst->order() == NCHW)
					{
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
					else if (dst->order() == NHWC)
					{
						for (size_t n = 0; n < num; n++)
						{
							int offset = n * 3 * height * width;
							for (size_t h = 0; h < height; h++)
							{
								int sub_offset = h * width * 3;
								for (size_t w = 0; w < width; w++)
								{
									int subsub_offset = w * 3;
									for (size_t c = 0; c < 3; c++)
									{
										dst_data[offset + sub_offset + subsub_offset + c] =
											Dtype((dst_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
									}
								}
							}
						}
					}
					else
					{
						NOT_IMPLEMENTED;
					}
				}
				else
				{
					Dtype* dst_data = dst->mutable_cpu_data();
					float means[] = { 127.5f };
					float var = 0.0078125f;

					for (size_t n = 0; n < num; n++)
					{
						int offset = n * height * width;
						for (size_t h = 0; h < height; h++)
						{
							int subsub_offset = h * width;
							for (size_t w = 0; w < width; w++)
							{
								dst_data[offset + subsub_offset + w] =
									Dtype((dst_data[offset + subsub_offset + w] - means[0]) * var);
							}
						}
					}
				}
			}



			/// <summary>
			/// 根据指定的矩形裁剪图像（类似于ROI），如果矩形超出图像边界，则用0填充对应像素值
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">裁剪后的tensor，包含了rect范围内的图像数据</param>
			/// <param name="rect">指定的矩形</param>
			template <typename Dtype, typename Rtype>
			static void safty_cut_cpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, rectangle<Rtype>* rect)
			{
				if (rect->x >= 0 && rect->y >= 0 && (rect->x + rect->w <= src->width()) && (rect->y + rect->h <= src->height()))
				{
					cut_border_cpu(src, dst, rect->y, (src->height() - rect->y - rect->h), rect->x, (src->width() - rect->x - rect->w));
				}
				else
				{
					int top = std::max(int(0), int(-1 * rect->y));
					int bottom = std::max(int(rect->y + rect->h - src->height()), int(0));
					int left = std::max(int(0), int(-1 * rect->x));
					int right = std::max(int(rect->x + rect->w - src->width()), int(0));
					std::shared_ptr<tensor<Dtype>> temp;
					if (src->order() == NCHW)
					{
						temp.reset(new tensor<Dtype>(
							std::vector<int>{src->num(), src->channels(), src->height() + top + bottom, src->width() + left + right},
							src->device(), src->order()));
					}
					else if (src->order() == NHWC)
					{
						temp.reset(new tensor<Dtype>(
							std::vector<int>{src->num(), src->height() + top + bottom, src->width() + left + right, src->channels()},
							src->device(), src->order()));
					}
					else
					{
						NOT_IMPLEMENTED;
					}

					make_border_cpu(src, temp, top, bottom, left, right, Border_Constant);
					cut_border_cpu(temp, dst, rect->y + top, temp->height() - rect->y - rect->h - top, rect->x + left, temp->width() - rect->x - rect->w - left);
				}
			}



			/// <summary>
			/// 根据指定的矩形裁剪图像（类似于ROI），如果矩形超出图像边界，则用0填充对应像素值
			/// </summary>
			/// <param name="src">包含原图像数据的tensor</param>
			/// <param name="dst">裁剪后的tensor，包含了rect范围内的图像数据</param>
			/// <param name="rect">指定的矩形</param>
			template <typename Dtype, typename Rtype>
			static void safty_cut_cpu(const tensor<Dtype>& src, tensor<Dtype>& dst, rectangle<Rtype>* rect)
			{
				if (rect->x >= 0 && rect->y >= 0 && (rect->x + rect->w <= src.width()) && (rect->y + rect->h <= src.height()))
				{
					cut_border_cpu(src, dst, rect->y, (src.height() - rect->y - rect->h), rect->x, (src.width() - rect->x - rect->w));
				}
				else
				{
					int top = std::max(0, -1 * rect->y);
					int bottom = std::max(rect->y + rect->h - src.height(), 0);
					int left = std::max(0, -1 * rect->x);
					int right = std::max(rect->x + rect->w - src.width(), 0);

					if (src.order() == NCHW)
					{
						tensor<Dtype>* temp = new tensor<Dtype>(
							std::vector<int>{src.num(), src.channels(), src.height() + top + bottom, src.width() + left + right},
							src.device(), src.type());
					}
					else if (src.order() == NHWC)
					{
						tensor<Dtype>* temp = new tensor<Dtype>(
							std::vector<int>{src.num(), src.height() + top + bottom, src.width() + left + right, src.channels()},
							src.device(), src.type());
					}
					else
					{
						NOT_IMPLEMENTED;
					}

					make_border_cpu(src, temp, top, bottom, left, right, Border_Constant);
					cut_border_cpu(temp, dst, rect->y + top, temp->height() - rect->y - rect->h - top, rect->x + left, temp->width() - rect->x - rect->w - left);
					delete temp;
				}
			}





		private:

#ifdef USE_OPENCV

			/// <summary>
			/// tensor转换为mat后用opencv显示
			/// </summary>
			/// <param name="dst">包含图像数据的tensor</param>
			template <typename Dtype>
			static void showimage(std::shared_ptr<tensor<Dtype>>& dst)
			{
				cv::Mat mat;
				tensor2mat(dst, mat);
				cv::Mat showmat;
				mat.convertTo(showmat, CV_8U);
				cv::imshow("test", showmat);
				cv::waitKey();
			}



			/// <summary>
			/// tensor转换为mat后用opencv显示
			/// </summary>
			/// <param name="dst">包含图像数据的tensor</param>
			template <typename Dtype>
			static void showimage(tensor<Dtype>& dst)
			{
				cv::Mat mat;
				tensor2mat(dst, mat);
				cv::Mat showmat;
				mat.convertTo(showmat, CV_8U);
				cv::imshow("test", showmat);
				cv::waitKey();
			}



			/// <summary>
			/// 根据数据类型获取对应的Mat类型
			/// </summary>
			/// <returns></returns>
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

			static int icvResize_Bilinear_8u_C1(const unsigned char* src, int srcstep, int swidth, int sheight,
				unsigned char* dst, int dststep, int dwidth, int dheight,
				int xmax,
				const CvResizeAlpha* xofs,
				const CvResizeAlpha* yofs,
				int* buf0, int* buf1)
			{
				int prev_sy0 = -1, prev_sy1 = -1;
				int k, dx, dy;

				srcstep /= sizeof(src[0]);
				dststep /= sizeof(dst[0]);

				for (dy = 0; dy < dheight; dy++, dst += dststep)
				{
					int fy = yofs[dy].ialpha, *swap_t;
					int sy0 = yofs[dy].idx, sy1 = sy0 + (fy > 0 && sy0 < sheight - 1);

					if (sy0 == prev_sy0 && sy1 == prev_sy1)
						k = 2;
					else if (sy0 == prev_sy1)
					{
						CV_SWAP(buf0, buf1, swap_t);
						k = 1;
					}
					else
						k = 0;

					for (; k < 2; k++)
					{
						int* _buf = k == 0 ? buf0 : buf1;
						const unsigned char* _src;
						int sy = k == 0 ? sy0 : sy1;
						if (k == 1 && sy1 == sy0)
						{
							memcpy(buf1, buf0, dwidth * sizeof(buf0[0]));
							continue;
						}

						_src = src + sy * srcstep;
						for (dx = 0; dx < xmax; dx++)
						{
							int sx = xofs[dx].idx;
							int fx = xofs[dx].ialpha;
							int t = _src[sx];
							_buf[dx] = ICV_WARP_MUL_ONE_8U(t) + fx * (_src[sx + 1] - t);
						}

						for (; dx < dwidth; dx++)
							_buf[dx] = ICV_WARP_MUL_ONE_8U(_src[xofs[dx].idx]);
					}

					prev_sy0 = sy0;
					prev_sy1 = sy1;

					if (sy0 == sy1)
						for (dx = 0; dx < dwidth; dx++)
							dst[dx] = (unsigned char)ICV_WARP_DESCALE_8U(ICV_WARP_MUL_ONE_8U(buf0[dx]));
					else
						for (dx = 0; dx < dwidth; dx++)
							dst[dx] = (unsigned char)ICV_WARP_DESCALE_8U(ICV_WARP_MUL_ONE_8U(buf0[dx]) +
								fy * (buf1[dx] - buf0[dx]));
				}

				return 1;
			}
		};
	}
}

#endif // !_TENSOR_OPERATION_CPU_HPP_