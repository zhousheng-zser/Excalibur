#ifndef _TENSOR_OPERATION_CPU_HPP_
#define _TENSOR_OPERATION_CPU_HPP_

#include "../../include/Primitives/tensor.hpp"
#include "../../include/Primitives/profiler.hpp"
#include "../../include/Primitives/simd_types.hpp"
#include "math_functions.hpp"
#include <algorithm>
#include <cstring>
#include <climits>
#include <cmath>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif // USE_OPENCV

#define PI 3.141592f
#define ICV_WARP_SHIFT          10
#define ICV_WARP_SHIFT2         15
#define ICV_SHIFT_DIFF          (ICV_WARP_SHIFT2-ICV_WARP_SHIFT)
#define ICV_WARP_MASK           ((1 << ICV_WARP_SHIFT) - 1)
#define ICV_WARP_MUL_ONE_8U(x)  ((x) << ICV_WARP_SHIFT)
#define ICV_WARP_DESCALE_8U(x)  CV_DESCALE((x), ICV_WARP_SHIFT*2)
#define CV_SWAP(a,b,t)          ((t) = (a), (a) = (b), (b) = (t))
#define CV_DESCALE(x,n)         (((x) + (1 << ((n)-1))) >> (n))

typedef struct CvResizeAlpha
{
	int idx;
	int ialpha;

}CvResizeAlpha;

extern const unsigned char LBPMAP[][256];

namespace glasssix
{
	namespace excalibur
	{
		enum interpolationType { Nearest, Bilinear, Cubic };

		enum borderType { Border_Constant, Border_Replicate };

		enum flipType { Channel_Wise, Width_Wise, Height_Wise, Center_Wise };

		enum thresholdType { binary, binary_inv, small_trunc, big_trunc, small_to_zero, big_to_zero };

		enum morphType { Dilate, Erode };

		///Rotation_Invariant(RI)
		enum lbpType { Native, RI, U2, RIU2, HF, LTP };

		template <typename Dtype>
		class point
		{
		public:
			Dtype x;
			Dtype y;

			point()
			{
				x = Dtype(0);
				y = Dtype(0);
			}

			point(Dtype x, Dtype y)
			{
				this->x = x;
				this->y = y;
			}

			point& operator=(const point& r)
			{
				if (this == &r)
				{
					return *this;
				}
				x = r.x;
				y = r.y;
				return *this;
			}

			point(const point& r)
			{
				x = r.x;
				y = r.y;
			}

			float distance(const point& r)
			{
				return sqrt((x - r.x) * (x - r.x) * 1.0f + (y - r.y) * (y - r.y) * 1.0f);
			}
		};

		template <typename Dtype>
		class rectangle
		{
		public:
			Dtype x;
			Dtype y;
			Dtype h;
			Dtype w;

			rectangle()
			{
				this->x = Dtype(0);
				this->y = Dtype(0);
				this->h = Dtype(0);
				this->w = Dtype(0);
			}

			rectangle(Dtype x, Dtype y, Dtype h, Dtype w)
			{
				CHECK_GE(h, 0);
				CHECK_GE(w, 0);
				this->x = x;
				this->y = y;
				this->h = h;
				this->w = w;
			}

			rectangle(point<Dtype> top_left, point<Dtype> bottom_right)
			{
				this->x = top_left.x;
				this->y = top_left.y;
				this->h = bottom_right.y - top_left.y;
				this->w = bottom_right.x - top_left.x;
				CHECK_GE(h, 0);
				CHECK_GE(w, 0);
			}

			rectangle& operator=(const rectangle& r)
			{
				if (this == &r)
				{
					return *this;
				}
				x = r.x;
				y = r.y;
				h = r.h;
				w = r.w;
				return *this;
			}

			rectangle(const rectangle& r)
			{
				x = r.x;
				y = r.y;
				h = r.h;
				w = r.w;
			}

			Dtype IoU(const rectangle r)
			{
				return Dtype(0);// Todo
			}

			point<Dtype> center()
			{
				return point<Dtype>(Dtype(x + w * 0.5f), Dtype(y + h * 0.5f));
			}
		};


		class color
		{
		public:
			unsigned char r;
			unsigned char g;
			unsigned char b;

			color()
			{
				r = (unsigned char)0;
				g = (unsigned char)0;
				b = (unsigned char)0;
			}

			template <typename Dtype>
			color(Dtype r, Dtype g, Dtype b)
			{
				this->r = (unsigned char)r;
				this->g = (unsigned char)g;
				this->b = (unsigned char)b;
			}


		};

		class tensor_operation_cpu
		{
		public:
			tensor_operation_cpu() {};
			~tensor_operation_cpu() {};

#ifdef USE_OPENCV

			/// <summary>
			/// transfer memory::tensor to mat
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new mat</param>
			template <typename Dtype>
			static void tensor2mat_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::vector<cv::Mat>& dst)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channel = src->channels();
				if (channel > 4)
				{
					LOG(ERROR) << "Too many channels.";
					return;
				}

				int width = src->width();
				int height = src->height();
				int type = get_cv_type<Dtype>();
				if (type < 0)
				{
					LOG(ERROR) << "Un-support data type.";
					return;
				}

				const Dtype* src_data = src->cpu_data();

				if (src->order() == memory::NCHW)
				{
					int src_offset = height * width;
					int* c_src_offset = new int[channel];

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * channel * height * width;
						cv::Mat temp = cv::Mat(height, width, CV_MAKETYPE(type, channel));						

						for (int c = 0; c < channel; c++)
						{
							c_src_offset[c] = c * src_offset;
						}

						for (int h = 0; h < height; h++)
						{
							int src_sub_offset = h * width;
							Dtype* temp_data = temp.ptr<Dtype>(h);
							
							for (int w = 0; w < width; w++)
							{
								for (int c = 0; c < channel; c++)
								{
									temp_data[w * channel + c] = src_data[n_offset + c_src_offset[c] + src_sub_offset + w];
								}
							}
						}

						dst.push_back(temp);
					}

					delete [] c_src_offset;
				}
				else if(src->order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * channel * height * width;
						cv::Mat temp = cv::Mat(height, width, CV_MAKETYPE(type, channel));						
						std::memcpy(temp.data, src_data + n_offset, channel * height * width * sizeof(Dtype));
						dst.push_back(temp);
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// transfer memory::tensor to mat
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new mat</param>
			template <typename Dtype>
			static void tensor2mat_cpu(const memory::tensor<Dtype>& src, std::vector<cv::Mat>& dst)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channel = src.channels();
				if (channel > 4)
				{
					LOG(ERROR) << "Too many channels.";
					return;
				}

				int width = src.width();
				int height = src.height();
				int type = get_cv_type<Dtype>();
				if (type < 0)
				{
					LOG(ERROR) << "Un-support data type.";
					return;
				}

				const Dtype* src_data = src.cpu_data();

				if (src.order() == memory::NCHW)
				{
					int src_offset = height * width;
					int* c_src_offset = new int[channel];

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * channel * height * width;
						cv::Mat temp = cv::Mat(height, width, CV_MAKETYPE(type, channel));						

						for (int c = 0; c < channel; c++)
						{
							c_src_offset[c] = c * src_offset;
						}

						for (int h = 0; h < height; h++)
						{
							int src_sub_offset = h * width;
							Dtype* temp_data = temp.ptr<Dtype>(h);
							
							for (int w = 0; w < width; w++)
							{
								for (int c = 0; c < channel; c++)
								{
									temp_data[w * channel + c] = src_data[n_offset + c_src_offset[c] + src_sub_offset + w];
								}
							}
						}

						dst.push_back(temp);
					}

					delete [] c_src_offset;
				}
				else if (src.order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * channel * height * width;
						cv::Mat temp = cv::Mat(height, width, CV_MAKETYPE(type, channel));						
						std::memcpy(temp.data, src_data + n_offset, channel * height * width * sizeof(Dtype));
						dst.push_back(temp);
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// transfer mat to memory::tensor
			/// </summary>
			/// <param name="src">original mat</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="order">order type of new memory::tensor: memory::NCHW / memory::NHWC(default)</param>
			template <typename Dtype>
			static void mat2tensor_cpu(const cv::Mat &src, std::shared_ptr<memory::tensor<Dtype>>& dst, orderType order = memory::NHWC)
			{
				if (src.data == NULL)
				{
					LOG(ERROR) << "No data.";
					return;
				}

				int channels = src.channels();
				int width = src.cols;
				int height = src.rows;
				int type_id = src.type() % 8;
				auto type_name = std::string(typeid(Dtype).name());

#ifdef _MSC_VER

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

#else

				if (type_id == 0)
				{
					if (type_name != std::string("h"))
					{
						LOG(ERROR) << "Un-matched data type.";
						return;
					}
				}
				else if (type_id == 1)
				{
					if (type_name != std::string("c"))
					{
						LOG(ERROR) << "Un-matched data type.";
						return;
					}
				}
				else if (type_id == 4)
				{
					if (type_name != std::string("i"))
					{
						LOG(ERROR) << "Un-matched data type.";
						return;
					}
				}
				else if (type_id == 5)
				{
					if (type_name != std::string("f"))
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
#endif // _MSC_VER

				if (order == memory::NCHW)
				{
					dst.reset(new memory::tensor<Dtype>(std::vector<int>{1, channels, height, width}, -1, memory::NCHW));
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

					delete [] c_dst_offset;
				}
				else if (order == memory::NHWC)
				{
					dst.reset(new memory::tensor<Dtype>(std::vector<int>{1, height, width, channels}, -1, memory::NHWC));
					Dtype* dst_data = dst->mutable_cpu_data();
					memcpy(dst_data, src.data, height * width * channels * sizeof(Dtype));
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// transfer mat to memory::tensor
			/// </summary>
			/// <param name="src">original mat</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="order">order type of new memory::tensor: memory::NCHW / memory::NHWC(default)</param>
			template <typename Dtype>
			static void mat2tensor_cpu(const cv::Mat &src, memory::tensor<Dtype>& dst, orderType order = memory::NHWC)
			{
				if (src.data == nullptr)
				{
					LOG(ERROR) << "No data.";
					return;
				}

				int channel = src.channels();
				int width = src.cols;
				int height = src.rows;
				int type_id = src.type() % 8;
				auto type_name = std::string(typeid(Dtype).name());

#ifdef _MSC_VER

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

#else

				if (type_id == 0)
				{
					if (type_name != std::string("h"))
					{
						LOG(ERROR) << "Un-matched data type.";
						return;
					}
				}
				else if (type_id == 1)
				{
					if (type_name != std::string("c"))
					{
						LOG(ERROR) << "Un-matched data type.";
						return;
					}
				}
				else if (type_id == 4)
				{
					if (type_name != std::string("i"))
					{
						LOG(ERROR) << "Un-matched data type.";
						return;
					}
				}
				else if (type_id == 5)
				{
					if (type_name != std::string("f"))
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
#endif // _MSC_VER

				if (order == memory::NCHW)
				{
					dst = memory::tensor<Dtype>(std::vector<int>{1, channel, height, width}, -1, memory::NCHW);
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

					delete [] c_dst_offset;
				}
				else if (order == memory::NHWC)
				{
					dst = memory::tensor<Dtype>(std::vector<int>{1, height, width, channel}, -1, memory::NHWC);
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
			/// convert between different datatype of memory::tensor
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename DtypeSRC, typename DtypeDST>
			static void type_converter_cpu(const std::shared_ptr<memory::tensor<DtypeSRC>> &src, std::shared_ptr<memory::tensor<DtypeDST>> &dst);



			/// <summary>
			/// convert between different datatype of memory::tensor
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename DtypeSRC, typename DtypeDST>
			static void type_converter_cpu(const memory::tensor<DtypeSRC> &src, memory::tensor<DtypeDST> &dst);



			/// <summary>
			/// convert memory::tensor(memory::NCHW) to memory::tensor(memory::NHWC)
			/// </summary>
			/// <param name="src">original memory::tensor(memory::NCHW)</param>
			/// <param name="dst">new memory::tensor(memory::NHWC)</param>
			template <typename Dtype>
			static void nchw2nhwc_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				CHECK_EQ(src->order(), memory::NCHW);
				int num = src->num();
				int height = src->height();
				int width = src->width();
				int channels = src->channels();
				int offset = height * width;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, height, width, channels}, src->device(), memory::NHWC));
				Dtype* dst_data = dst_temp->mutable_cpu_data();
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

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// convert memory::tensor(memory::NCHW) to memory::tensor(memory::NHWC)
			/// </summary>
			/// <param name="src">original memory::tensor(memory::NCHW)</param>
			/// <param name="dst">new memory::tensor(memory::NHWC)</param>
			template <typename Dtype>
			static void nchw2nhwc_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				CHECK_EQ(src.order(), memory::NCHW);
				int num = src.num();
				int height = src.height();
				int width = src.width();
				int channels = src.channels();
				int offset = height * width;

				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(std::vector<int>{num, height, width, channels}, src.device(), memory::NHWC);
				Dtype* dst_data = dst_temp.mutable_cpu_data();
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

				dst = dst_temp.clone();
			}



			/// <summary>
			/// convert memory::tensor(memory::NHWC) to memory::tensor(memory::NCHW)
			/// </summary>
			/// <param name="src">original memory::tensor(memory::NHWC)</param>
			/// <param name="dst">new memory::tensor(memory::NCHW)</param>
			template <typename Dtype>
			static void nhwc2nchw_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				CHECK_EQ(src->order(), memory::NHWC);
				int num = src->num();
				int height = src->height();
				int width = src->width();
				int channels = src->channels();
				int offset = height * width;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, height, width}, src->device(), memory::NCHW));
				Dtype* dst_data = dst_temp->mutable_cpu_data();
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

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// convert memory::tensor(memory::NHWC) to memory::tensor(memory::NCHW)
			/// </summary>
			/// <param name="src">original memory::tensor(memory::NHWC)</param>
			/// <param name="dst">new memory::tensor(memory::NCHW)</param>
			template <typename Dtype>
			static void nhwc2nchw_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				CHECK_EQ(src.order(), memory::NHWC);
				int num = src.num();
				int height = src.height();
				int width = src.width();
				int channels = src.channels();
				int offset = height * width;

				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(std::vector<int>{num, channels, height, width}, src.device(), memory::NCHW);
				Dtype* dst_data = dst_temp.mutable_cpu_data();
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

				dst = dst_temp.clone();
			}


			
			/// <summary>
			/// resize image data
			/// </summary>
			/// <param name="src">memory::tensor of image with original size(height and width)</param>
			/// <param name="dst">memory::tensor of image with new size</param>
			/// <param name="dst_height">new height</param>
			/// <param name="dst_width">new width</param>
			/// <param name="type">interpolationType: Nearest / Bilinear(default)</param>
			template <typename Dtype>
			static void resize_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst,
				int dst_height, int dst_width, interpolationType type = Bilinear)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (dst_height * dst_width <= 0)
				{
					LOG(ERROR) << "Illegal input size.";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int src_offset = height * width;
				int dst_offset = dst_height * dst_width;
				int src_num_offset = channels * height * width;
				int dst_num_offset = channels * dst_height * dst_width;
				unsigned maxIndex = height * width * channels - 1;

				if (dst_width == width && dst_height == height)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				if (src->order() == memory::NCHW)
				{					
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
					Dtype* dst_data = dst_temp->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					auto name = typeid(Dtype).name();

#ifdef _MSC_VER
					if (std::string("unsigned char") == std::string(name))
#else
					if (std::string("h") == std::string(name))
#endif
					{
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
							{
								sx = 0;
								fx_1024x = 0;
							}

							if (sx >= width - 1)
							{
								fx_1024x = 0;
								sx = width - 1;

								if (xmax >= dst_width)
								{
									xmax = dx;
								}
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
							{
								sy = 0;
								fy_1024x = 0;
							}

							yofs[dy].idx = sy;
							yofs[dy].ialpha = fy_1024x;
						}

						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;

							for (int ch = 0; ch < channels; ++ch)
							{
								int src_channel_offset = ch * src_offset;
								int dst_channel_offset = ch * dst_offset;

								icvResize_Bilinear_8u_C1((const unsigned char*)&src_data[src_n_offset + src_channel_offset], width * sizeof(unsigned char), width, height, (unsigned char*)&dst_data[dst_n_offset + dst_channel_offset],
									dst_width * sizeof(unsigned char), dst_width, dst_height, xmax, xofs, yofs, buf0, buf1);
							}
						}

						free(temp_buf);
					}
					else
					{
						float width_ratio = (float)width / dst_width;
						float height_ratio = (float)height / dst_height;
						float beta = 0.5f;

#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int row = 0; row < dst_height; ++row)
						{
							float yf = row * height_ratio + beta;
							int y = (int)yf;
							float ydiff = yf - y;

							int src_pos1 = y * width;
							int dst_pos1 = row * dst_width;

							for (int col = 0; col < dst_width; ++col)
							{
								float xf = col * width_ratio + beta;
								int x = (int)xf;
								float xdiff = xf - x;

								int src_pos2 = src_pos1 + x;
								int dst_pos2 = dst_pos1 + col;

								for (int n = 0; n < num; n++)
								{
									int src_n_offset = n * src_num_offset;
									int dst_n_offset = n * dst_num_offset;

									for (int ch = 0; ch < channels; ++ch)
									{
										int src_pos3 = src_pos2 + ch * src_offset;
										int dst_pos3 = dst_pos2 + ch * dst_offset;

										if (type == Nearest)
										{
											dst_data[dst_n_offset + dst_pos3] = src_data[src_n_offset + src_pos3];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos3), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos3 + 1), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos3 + width), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos3 + (width + 1)), maxIndex);
											Dtype A = src_data[src_n_offset + indexA];
											Dtype B = src_data[src_n_offset + indexB];
											Dtype C = src_data[src_n_offset + indexC];
											Dtype D = src_data[src_n_offset + indexD];

											dst_data[dst_n_offset + dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else if (src->order() == memory::NHWC)
				{
					float width_ratio = (float)width / dst_width;
					float height_ratio = (float)height / dst_height;
					float beta = 0.5f;

					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst_temp->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

#ifdef _OPENMP
#pragma omp parallel for
#endif
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

							for (int n = 0; n < num; n++)
							{
								int src_n_offset = n * src_num_offset;
								int dst_n_offset = n * dst_num_offset;

								for (int ch = 0; ch < channels; ++ch)
								{
									int src_pos3 = src_pos2 + ch;
									int dst_pos3 = dst_pos2 + ch;

									if (type == Nearest)
									{
										dst_data[dst_n_offset + dst_pos3] = src_data[src_n_offset + src_pos3];
									}
									else if (type == Bilinear)
									{
										unsigned indexA = std::min(unsigned(src_pos3), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos3 + channels), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos3 + width * channels), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos3 + (width + 1) * channels), maxIndex);
										Dtype A = src_data[src_n_offset + indexA];
										Dtype B = src_data[src_n_offset + indexB];
										Dtype C = src_data[src_n_offset + indexC];
										Dtype D = src_data[src_n_offset + indexD];

										dst_data[dst_n_offset + dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// resize image data
			/// </summary>
			/// <param name="src">memory::tensor of image with original size(height and width)</param>
			/// <param name="dst">memory::tensor of image with new size</param>
			/// <param name="dst_height">new height</param>
			/// <param name="dst_width">new width</param>
			/// <param name="type">interpolationType: Nearest / Bilinear(default)</param>
			template <typename Dtype>
			static void resize_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst,
				int dst_height, int dst_width, interpolationType type = Bilinear)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (dst_height * dst_width <= 0)
				{
					LOG(ERROR) << "Illegal input size.";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;
				int dst_offset = dst_height * dst_width;
				int src_num_offset = channels * height * width;
				int dst_num_offset = channels * dst_height * dst_width;
				unsigned maxIndex = height * width * channels - 1;

				if (dst_width == width && dst_height == height)
				{
					dst = src.clone();
					return;
				}

				memory::tensor<Dtype> dst_temp;
				if (src.order() == memory::NCHW)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src.device(), src.order());
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					auto name = typeid(Dtype).name();
#ifdef _MSC_VER
					if (std::string("unsigned char") == std::string(name))
#else
					if (std::string("h") == std::string(name))
#endif
					{
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
							{
								sx = 0;
								fx_1024x = 0;
							}

							if (sx >= width - 1)
							{
								fx_1024x = 0;
								sx = width - 1;

								if (xmax >= dst_width)
								{
									xmax = dx;
								}
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
							{
								sy = 0;
								fy_1024x = 0;
							}

							yofs[dy].idx = sy;
							yofs[dy].ialpha = fy_1024x;
						}

						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;

							for (int ch = 0; ch < channels; ++ch)
							{
								int src_channel_offset = ch * src_offset;
								int dst_channel_offset = ch * dst_offset;

								icvResize_Bilinear_8u_C1((const unsigned char*)&src_data[src_n_offset + src_channel_offset], width * sizeof(unsigned char), width, height, (unsigned char*)&dst_data[dst_n_offset + dst_channel_offset],
									dst_width * sizeof(unsigned char), dst_width, dst_height, xmax, xofs, yofs, buf0, buf1);
							}
						}

						free(temp_buf);
					}
					else
					{
						float width_ratio = (float)width / dst_width;
						float height_ratio = (float)height / dst_height;
						float beta = 0.5f;

#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int row = 0; row < dst_height; ++row)
						{
							float yf = row * height_ratio + beta;
							int y = (int)yf;
							float ydiff = yf - y;

							int src_pos1 = y * width;
							int dst_pos1 = row * dst_width;

							for (int col = 0; col < dst_width; ++col)
							{
								float xf = col * width_ratio + beta;
								int x = (int)xf;
								float xdiff = xf - x;

								int src_pos2 = src_pos1 + x;
								int dst_pos2 = dst_pos1 + col;

								for (int n = 0; n < num; n++)
								{
									int src_n_offset = n * src_num_offset;
									int dst_n_offset = n * dst_num_offset;

									for (int ch = 0; ch < channels; ++ch)
									{
										int src_pos3 = src_pos2 + ch * src_offset;
										int dst_pos3 = dst_pos2 + ch * dst_offset;

										if (type == Nearest)
										{
											dst_data[dst_n_offset + dst_pos3] = src_data[src_n_offset + src_pos3];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos3), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos3 + channels), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos3 + width * channels), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos3 + (width + 1) * channels), maxIndex);
											Dtype A = src_data[src_n_offset + indexA];
											Dtype B = src_data[src_n_offset + indexB];
											Dtype C = src_data[src_n_offset + indexC];
											Dtype D = src_data[src_n_offset + indexD];

											dst_data[dst_n_offset + dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else if (src.order() == memory::NHWC)
				{
					float width_ratio = (float)width / dst_width;
					float height_ratio = (float)height / dst_height;
					float beta = 0.5f;

					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src.device(), src.order());
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

#ifdef _OPENMP
#pragma omp parallel for
#endif
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

							for (int n = 0; n < num; n++)
							{
								int src_n_offset = n * src_num_offset;
								int dst_n_offset = n * dst_num_offset;

								for (int ch = 0; ch < channels; ++ch)
								{
									int src_pos3 = src_pos2 + ch;
									int dst_pos3 = dst_pos2 + ch;

									if (type == Nearest)
									{
										dst_data[dst_n_offset + dst_pos3] = src_data[src_n_offset + src_pos3];
									}
									else if (type == Bilinear)
									{
										unsigned indexA = std::min(unsigned(src_pos3), maxIndex);
										unsigned indexB = std::min(unsigned(src_pos3 + channels), maxIndex);
										unsigned indexC = std::min(unsigned(src_pos3 + width * channels), maxIndex);
										unsigned indexD = std::min(unsigned(src_pos3 + (width + 1) * channels), maxIndex);
										Dtype A = src_data[src_n_offset + indexA];
										Dtype B = src_data[src_n_offset + indexB];
										Dtype C = src_data[src_n_offset + indexC];
										Dtype D = src_data[src_n_offset + indexD];

										dst_data[dst_n_offset + dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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

				dst = dst_temp.clone();
			}



			/// <summary>
			/// rotate around center point, height and width will be changed
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="theta">rotation angle, anti-clockwise is positive</param>
			/// <param name="dst_height">new height after rotate</param>
			/// <param name="dst_width">new width after rotate</param>
			/// <param name="fill_pixel_value">pixel value to fill in the blank area, zero by default</param>
			/// <param name="type">interpolationType: Nearest / Bilinear(default)</param>
			template <typename Dtype>
			static void rotate_with_center_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst,
				float theta, int &dst_height, int &dst_width, Dtype fill_pixel_value = 0, interpolationType type = Bilinear)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (fabs(theta) <= 1e-6)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int src_offset = height * width;
				int src_num_offset = channels * height * width;
				unsigned maxIndex = height * width * channels - 1;

				float rad = -1 * theta * (PI / 180);//anti-clockwise is positive
				float cosa = cos(rad);
				float sina = sin(rad);

				dst_width = (int)(width * abs(cosa) + height * abs(sina));
				dst_height = (int)(width * abs(sina) + height * abs(cosa));
				int dst_offset = dst_height * dst_width;
				int dst_num_offset = channels * dst_height * dst_width;

				float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
				float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				if (src->order() == memory::NCHW)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
					Dtype* dst_data = dst_temp->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										dst_data[dst_pos2] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[dst_n_offset + dst_pos2] = src_data[src_n_offset + src_pos1];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos1), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos1 + 1), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos1 + width), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos1 + width + 1), maxIndex);
											Dtype A = src_data[src_n_offset + indexA];
											Dtype B = src_data[src_n_offset + indexB];
											Dtype C = src_data[src_n_offset + indexC];
											Dtype D = src_data[src_n_offset + indexD];

											dst_data[dst_n_offset + dst_pos2] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else if (src->order() == memory::NHWC)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst_temp->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

#ifdef _OPENMP
#pragma omp parallel for
#endif
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

							for (int n = 0; n < num; n++)
							{
								int src_n_offset = n * src_num_offset;
								int dst_n_offset = n * dst_num_offset;

								for (int ch = 0; ch < channels; ++ch)
								{
									int src_pos2 = src_pos1 + ch;
									int dst_pos3 = dst_pos2 + ch;

									if (x >= width || x < 0 || y >= height || y < 0)
									{
										dst_data[dst_n_offset + dst_pos3] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[dst_n_offset + dst_pos3] = src_data[src_n_offset + src_pos2];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos2), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos2 + channels), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos2 + width * channels), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
											Dtype A = src_data[src_n_offset + indexA];
											Dtype B = src_data[src_n_offset + indexB];
											Dtype C = src_data[src_n_offset + indexC];
											Dtype D = src_data[src_n_offset + indexD];

											dst_data[dst_n_offset + dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// rotate around center point, height and width will be changed
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="theta">rotation angle, anti-clockwise is positive</param>
			/// <param name="dst_height">new height after rotate</param>
			/// <param name="dst_width">new width after rotate</param>
			/// <param name="fill_pixel_value">pixel value to fill in the blank area, zero by default</param>
			/// <param name="type">interpolationType: Nearest / Bilinear(default)</param>
			template <typename Dtype>
			static void rotate_with_center_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype>& dst,
				float theta, int &dst_height, int &dst_width, Dtype fill_pixel_value = 0, interpolationType type = Bilinear)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (fabs(theta) <= 1e-6)
				{
					dst = src.clone();
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;
				int src_num_offset = channels * height * width;
				unsigned maxIndex = height * width * channels - 1;

				float rad = -1 * theta * (PI / 180);//anti-clockwise is positive
				float cosa = cos(rad);
				float sina = sin(rad);

				dst_width = (int)(width * abs(cosa) + height * abs(sina));
				dst_height = (int)(width * abs(sina) + height * abs(cosa));
				int dst_offset = dst_height * dst_width;
				int dst_num_offset = channels * dst_height * dst_width;

				float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
				float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

				memory::tensor<Dtype> dst_temp;
				if (src.order() == memory::NCHW)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src.device(), src.order());
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										dst_data[dst_n_offset + dst_pos2] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[dst_n_offset + dst_pos2] = src_data[src_n_offset + src_pos1];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos1), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos1 + 1), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos1 + width), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos1 + width + 1), maxIndex);
											Dtype A = src_data[src_n_offset + indexA];
											Dtype B = src_data[src_n_offset + indexB];
											Dtype C = src_data[src_n_offset + indexC];
											Dtype D = src_data[src_n_offset + indexD];

											dst_data[dst_n_offset + dst_pos2] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else if (src.order() == memory::NHWC)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src.device(), src.order());
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

#ifdef _OPENMP
#pragma omp parallel for
#endif
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

							for (int n = 0; n < num; n++)
							{
								int src_n_offset = n * src_num_offset;
								int dst_n_offset = n * dst_num_offset;

								for (int ch = 0; ch < channels; ++ch)
								{
									int src_pos2 = src_pos1 + ch;
									int dst_pos3 = dst_pos2 + ch;

									if (x >= width || x < 0 || y >= height || y < 0)
									{
										dst_data[dst_n_offset + dst_pos3] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[dst_n_offset + dst_pos3] = src_data[src_n_offset + src_pos2];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos2), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos2 + channels), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos2 + width * channels), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
											Dtype A = src_data[src_n_offset + indexA];
											Dtype B = src_data[src_n_offset + indexB];
											Dtype C = src_data[src_n_offset + indexC];
											Dtype D = src_data[src_n_offset + indexD];

											dst_data[dst_n_offset + dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// rotate around any point, height and width will be constant
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="center">rotation center</param>
			/// <param name="theta">rotation angle, anti-clockwise is positive</param>
			/// <param name="scale">ratio of scale, 1 by default</param>
			/// <param name="fill_pixel_value">pixel value to fill in the blank area, zero by default</param>
			/// <param name="type">interpolationType: Nearest / Bilinear(default)</param>
			template <typename Dtype, typename Ptype>
			static void rotate_with_points_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst,
				const point<Ptype> &center, float theta, float scale = 1.0f, Dtype fill_pixel_value = 0, interpolationType type = Bilinear)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (fabs(theta) <= 1e-6)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;
				unsigned maxIndex = height * width * channels - 1;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst_temp->mutable_cpu_data();

				double rad = theta*(PI / 180);
				double cosa = cos(rad);
				double sina = sin(rad);

				double a = scale * cosa;
				double b = scale * sina;
				std::vector<std::vector<double> > M;
				std::vector<std::vector<double> > reverse_M;
				M.resize(3);

				for (int i = 0; i < M.size(); i++)
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

				if (src->order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										dst_data[n_offset + dst_index] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[n_offset + dst_index] = src_data[n_offset + src_index];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_index), maxIndex);
											unsigned indexB = std::min(unsigned(src_index + 1), maxIndex);
											unsigned indexC = std::min(unsigned(src_index + width), maxIndex);
											unsigned indexD = std::min(unsigned(src_index + width + 1), maxIndex);
											Dtype A = src_data[n_offset + indexA];
											Dtype B = src_data[n_offset + indexB];
											Dtype C = src_data[n_offset + indexC];
											Dtype D = src_data[n_offset + indexD];

											dst_data[n_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else if (src->order() == memory::NHWC)
				{
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

							for (int n = 0; n < num; n++)
							{
								int n_offset = n * num_offset;
								for (int ch = 0; ch < channels; ++ch)
								{
									int src_pos2 = src_pos1 + ch;
									int dst_pos3 = dst_pos2 + ch;

									if (x < 0 || x >= width || y < 0 || y >= height)
									{
										dst_data[n_offset + dst_pos3] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[n_offset + dst_pos3] = src_data[n_offset + src_pos2];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos2), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos2 + channels), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos2 + width * channels), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
											Dtype A = src_data[n_offset + indexA];
											Dtype B = src_data[n_offset + indexB];
											Dtype C = src_data[n_offset + indexC];
											Dtype D = src_data[n_offset + indexD];

											dst_data[n_offset + dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// rotate around any point, height and width will be constant
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="center">rotation center</param>
			/// <param name="theta">rotation angle, anti-clockwise is positive</param>
			/// <param name="scale">ratio of scale, 1 by default</param>
			/// <param name="fill_pixel_value">pixel value to fill in the blank area, zero by default</param>
			/// <param name="type">interpolationType: Nearest / Bilinear(default)</param>
			template <typename Dtype, typename Ptype>
			static void rotate_with_points_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst,
				const point<Ptype> &center, float theta, float scale = 1.0f, Dtype fill_pixel_value = 0, interpolationType type = Bilinear)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (fabs(theta) <= 1e-6)
				{
					dst = src.clone();
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;
				unsigned maxIndex = height * width * channels - 1;

				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());
				const Dtype* src_data = src.cpu_data();
				Dtype* dst_data = dst_temp.mutable_cpu_data();

				double rad = theta*(PI / 180);
				double cosa = cos(rad);
				double sina = sin(rad);

				double a = scale * cosa;
				double b = scale * sina;

				std::vector<std::vector<double> > M;
				std::vector<std::vector<double> > reverse_M;
				M.resize(3);

				for (int i = 0; i < M.size(); i++)
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

				if (src.order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										dst_data[n_offset + dst_index] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[n_offset + dst_index] = src_data[n_offset + src_index];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_index), maxIndex);
											unsigned indexB = std::min(unsigned(src_index + 1), maxIndex);
											unsigned indexC = std::min(unsigned(src_index + width), maxIndex);
											unsigned indexD = std::min(unsigned(src_index + width + 1), maxIndex);
											Dtype A = src_data[n_offset + indexA];
											Dtype B = src_data[n_offset + indexB];
											Dtype C = src_data[n_offset + indexC];
											Dtype D = src_data[n_offset + indexD];

											dst_data[n_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else if (src.order() == memory::NHWC)
				{
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

							for (int n = 0; n < num; n++)
							{
								int n_offset = n * num_offset;

								for (int ch = 0; ch < channels; ++ch)
								{
									int src_pos2 = src_pos1 + ch;
									int dst_pos3 = dst_pos2 + ch;

									if (x < 0 || x >= width || y < 0 || y >= height)
									{
										dst_data[n_offset + dst_pos3] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[n_offset + dst_pos3] = src_data[n_offset + src_pos2];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos2), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos2 + channels), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos2 + width * channels), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
											Dtype A = src_data[n_offset + indexA];
											Dtype B = src_data[n_offset + indexB];
											Dtype C = src_data[n_offset + indexC];
											Dtype D = src_data[n_offset + indexD];

											dst_data[n_offset + dst_pos3] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// draw rectangle
			/// </summary>
			/// <param name="dst">memory::tensor with image data</param>
			/// <param name="rect">rectangle to be drawn</param>
			/// <param name="color">color of lines</param>
			/// <param name="thickness">thickness of lines, 2 pixels by default</param>
			template <typename Dtype, typename Rtype>
			static void draw_rectangle_cpu(std::shared_ptr<memory::tensor<Dtype>>& dst, rectangle<Rtype> rect, color color, int thickness = 2)
			{
				if (dst->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = dst->num();
				int channels = dst->channels();
				int height = dst->height();
				int width = dst->width();
				int src_offset = height * width;
				int num_offset = channels * height * width;
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
					Dtype fill_color = Dtype(color.g / 3 + color.b / 3 + color.r / 3);

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

						//top
						for (int row = rect.y; row < rect.y + thickness; ++row)
						{
							Dtype* row_data = dst_data + n_offset + row * width;
							for (int col = rect.x; col < rect.x + rect.w; ++col)
							{
								row_data[col] = fill_color;
							}
						}

						//bottom
						for (int row = rect.y + rect.h - thickness; row < rect.y + rect.h; ++row)
						{
							Dtype* row_data = dst_data + n_offset + row * width;
							for (int col = rect.x; col < rect.x + rect.w; ++col)
							{
								row_data[col] = fill_color;
							}
						}

						//2-sides
						for (int row = rect.y; row < rect.y + rect.h; ++row)
						{
							Dtype* row_data = dst_data + n_offset + row * width;

							//left-side
							for (int col = rect.x; col < rect.x + thickness; ++col)
							{
								row_data[col] = fill_color;
							}

							//right-side
							for (int col = rect.x + rect.w - thickness; col < rect.x + rect.w; ++col)
							{
								row_data[col] = fill_color;
							}
						}
					}
				}
				else if (channels == 3 || channels == 4)
				{
					Dtype* fill_color = new Dtype[3];
					fill_color[0] = color.r;
					fill_color[1] = color.g;
					fill_color[2] = color.b;

					if (dst->order() == memory::NCHW)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < 3; ++ch)
							{
								int src_channel_offset = ch * src_offset;
								//top
								for (int row = rect.y; row < rect.y + thickness; ++row)
								{
									Dtype* row_data = dst_data + n_offset + src_channel_offset + row * width;
									for (int col = rect.x; col < rect.x + rect.w; ++col)
									{
										row_data[col] = fill_color[ch];
									}
								}

								//bottom
								for (int row = rect.y + rect.h - thickness; row < rect.y + rect.h; ++row)
								{
									Dtype* row_data = dst_data + n_offset + src_channel_offset + row * width;
									for (int col = rect.x; col < rect.x + rect.w; ++col)
									{
										row_data[col] = fill_color[ch];
									}
								}

								//2-sides
								for (int row = rect.y; row < rect.y + rect.h; ++row)
								{
									Dtype* row_data = dst_data + n_offset + src_channel_offset + row * width;

									//left-side
									for (int col = rect.x; col < rect.x + thickness; ++col)
									{
										row_data[col] = fill_color[ch];
									}

									//right-side
									for (int col = rect.x + rect.w - thickness; col < rect.x + rect.w; ++col)
									{
										row_data[col] = fill_color[ch];
									}
								}
							}
						}
					}
					else if (dst->order() == memory::NHWC)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;

							//top
							for (int row = rect.y; row < rect.y + thickness; ++row)
							{
								int pos1 = (row * width + rect.x) * channels;

								for (int col = 0; col < rect.w; ++col)
								{
									int pos2 = pos1 + col * channels;
									for (int ch = 0; ch < 3; ++ch)
									{
										dst_data[n_offset + pos2 + ch] = fill_color[ch];
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
										dst_data[n_offset + pos2 + ch] = fill_color[ch];
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
										dst_data[n_offset + left_pos2 + ch] = fill_color[ch];
										dst_data[n_offset + right_pos2 + ch] = fill_color[ch];
									}
								}
							}
						}
					}
					else
					{
						NOT_IMPLEMENTED;
					}

					delete[] fill_color;
				}
				else
				{
					LOG(FATAL) << "Illegal channel numbers. Return without any changes.";
				}
			}



			/// <summary>
			/// draw rectangle
			/// </summary>
			/// <param name="dst">memory::tensor with image data</param>
			/// <param name="rect">rectangle to be drawn</param>
			/// <param name="color">color of lines</param>
			/// <param name="thickness">thickness of lines, 2 pixels by default</param>
			template <typename Dtype, typename Rtype>
			static void draw_rectangle_cpu(memory::tensor<Dtype>& dst, rectangle<Rtype> rect, color color, int thickness = 2)
			{
				if (dst.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = dst.num();
				int channels = dst.channels();
				int height = dst.height();
				int width = dst.width();
				int src_offset = height * width;
				int num_offset = channels * height * width;
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
					Dtype fill_color = Dtype(color.g / 3 + color.b / 3 + color.r / 3);

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

						//top
						for (int row = rect.y; row < rect.y + thickness; ++row)
						{
							Dtype* row_data = dst_data + n_offset + row * width;
							for (int col = rect.x; col < rect.x + rect.w; ++col)
							{
								row_data[col] = fill_color;
							}
						}

						//bottom
						for (int row = rect.y + rect.h - thickness; row < rect.y + rect.h; ++row)
						{
							Dtype* row_data = dst_data + n_offset + row * width;
							for (int col = rect.x; col < rect.x + rect.w; ++col)
							{
								row_data[col] = fill_color;
							}
						}

						//2-sides
						for (int row = rect.y; row < rect.y + rect.h; ++row)
						{
							Dtype* row_data = dst_data + n_offset + row * width;

							//left-side
							for (int col = rect.x; col < rect.x + thickness; ++col)
							{
								row_data[col] = fill_color;
							}

							//right-side
							for (int col = rect.x + rect.w - thickness; col < rect.x + rect.w; ++col)
							{
								row_data[col] = fill_color;
							}
						}
					}
				}
				else if (channels == 3 || channels == 4)
				{
					Dtype* fill_color = new Dtype[3];
					fill_color[0] = color.r;
					fill_color[1] = color.g;
					fill_color[2] = color.b;

					if (dst.order() == memory::NCHW)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < 3; ++ch)
							{
								int src_channel_offset = ch * src_offset;

								//top
								for (int row = rect.y; row < rect.y + thickness; ++row)
								{
									Dtype* row_data = dst_data + n_offset + src_channel_offset + row * width;
									for (int col = rect.x; col < rect.x + rect.w; ++col)
									{
										row_data[col] = fill_color[ch];
									}
								}

								//bottom
								for (int row = rect.y + rect.h - thickness; row < rect.y + rect.h; ++row)
								{
									Dtype* row_data = dst_data + n_offset + src_channel_offset + row * width;
									for (int col = rect.x; col < rect.x + rect.w; ++col)
									{
										row_data[col] = fill_color[ch];
									}
								}

								//2-sides
								for (int row = rect.y; row < rect.y + rect.h; ++row)
								{
									Dtype* row_data = dst_data + n_offset + src_channel_offset + row * width;

									//left-side
									for (int col = rect.x; col < rect.x + thickness; ++col)
									{
										row_data[col] = fill_color[ch];
									}

									//right-side
									for (int col = rect.x + rect.w - thickness; col < rect.x + rect.w; ++col)
									{
										row_data[col] = fill_color[ch];
									}
								}
							}
						}
					}
					else if (dst.order() == memory::NHWC)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;

							//top
							for (int row = rect.y; row < rect.y + thickness; ++row)
							{
								int pos1 = (row * width + rect.x) * channels;

								for (int col = 0; col < rect.w; ++col)
								{
									int pos2 = pos1 + col * channels;
									for (int ch = 0; ch < 3; ++ch)
									{
										dst_data[n_offset + pos2 + ch] = fill_color[ch];
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
										dst_data[n_offset + pos2 + ch] = fill_color[ch];
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
										dst_data[n_offset + left_pos2 + ch] = fill_color[ch];
										dst_data[n_offset + right_pos2 + ch] = fill_color[ch];
									}
								}
							}
						}
					}
					else
					{
						NOT_IMPLEMENTED;
					}

					delete[] fill_color;
				}
				else
				{
					LOG(FATAL) << "Illegal channel numbers. Return without any changes.";
				}
			}



			/// <summary>
			/// draw circle
			/// </summary>
			/// <param name="dst">memory::tensor with image data</param>
			/// <param name="center">circle center</param>
			/// <param name="radius">radius</param>
			/// <param name="color">color of lines</param>
			/// <param name="thickness">thickness of lines, 2 pixels by default</param>
			template <typename Dtype, typename Rtype>
			static void draw_circle_cpu(std::shared_ptr<memory::tensor<Dtype>>& dst, point<Rtype> center, int radius, color color, int thickness = 2)
			{
				if (dst->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = dst->num();
				int channels = dst->channels();
				int height = dst->height();
				int width = dst->width();
				int offset = height * width;
				int num_offset = channels * height * width;
				Dtype* dst_data = dst->mutable_cpu_data();

				if (center.x < radius || center.x >= width - radius || center.y < radius || center.y >= height - radius)
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
								Dtype fill_color = Dtype(color.g / 3 + color.b / 3 + color.r / 3);

								for (int n = 0; n < num; n++)
								{
									int n_offset = n * num_offset;
									Dtype *tptr0 = dst_data + n_offset + y11 * width;
									Dtype *tptr1 = dst_data + n_offset + y12 * width;

									for (int i = x11; i < x11 + 2; i++)
									{
										tptr0[i] = fill_color;
										tptr1[i] = fill_color;
									}

									for (int i = x12; i < x12 + 2; i++)
									{
										tptr0[i] = fill_color;
										tptr1[i] = fill_color;
									}

									tptr0 = dst_data + n_offset + y21 * width;
									tptr1 = dst_data + n_offset + y22 * width;

									for (int i = x21; i < x21 + 2; i++)
									{
										tptr0[i] = fill_color;
										tptr1[i] = fill_color;
									}

									for (int i = x22; i < x22 + 2; i++)
									{
										tptr0[i] = fill_color;
										tptr1[i] = fill_color;
									}
								}
							}
							else if (channels == 3 || channels == 4)
							{
								Dtype* fill_color = new Dtype[3];
								fill_color[0] = color.r;
								fill_color[1] = color.g;
								fill_color[2] = color.b;

								if (dst->order() == memory::NCHW)
								{
									for (int n = 0; n < num; n++)
									{
										int n_offset = n * num_offset;
										for (int ch = 0; ch < 3; ++ch)
										{
											int channel_offset = ch * offset;

											Dtype *tptr0 = dst_data + n_offset + channel_offset + y11 * width;
											Dtype *tptr1 = dst_data + n_offset + channel_offset + y12 * width;

											for (int i = x11; i < x11 + 2; i++)
											{
												tptr0[i] = fill_color[ch];
												tptr1[i] = fill_color[ch];
											}

											for (int i = x12; i < x12 + 2; i++)
											{
												tptr0[i] = fill_color[ch];
												tptr1[i] = fill_color[ch];
											}

											tptr0 = dst_data + n_offset + channel_offset + y21 * width;
											tptr1 = dst_data + n_offset + channel_offset + y22 * width;

											for (int i = x21; i < x21 + 2; i++)
											{
												tptr0[i] = fill_color[ch];
												tptr1[i] = fill_color[ch];
											}

											for (int i = x22; i < x22 + 2; i++)
											{
												tptr0[i] = fill_color[ch];
												tptr1[i] = fill_color[ch];
											}
										}
									}
								}
								else if (dst->order() == memory::NHWC)
								{
									int pos_x11_y11 = (y11 * width + x11) * 3;
									int pos_x11_y12 = (y12 * width + x11) * 3;
									int pos_x12_y11 = (y11 * width + x12) * 3;
									int pos_x12_y12 = (y12 * width + x12) * 3;

									int pos_x21_y21 = (y21 * width + x21) * 3;
									int pos_x21_y22 = (y22 * width + x21) * 3;
									int pos_x22_y21 = (y21 * width + x22) * 3;
									int pos_x22_y22 = (y22 * width + x22) * 3;

									for (int n = 0; n < num; n++)
									{
										int n_offset = n * num_offset;
										for (int col = 0; col < 2; ++col)//fill neighboring 2 pixels
										{
											for (int ch = 0; ch < 3; ++ch)
											{
												dst_data[n_offset + pos_x11_y11 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x11_y12 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x12_y11 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x12_y12 + col * 3 + ch] = fill_color[ch];
												
												dst_data[n_offset + pos_x21_y21 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x21_y22 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x22_y21 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x22_y22 + col * 3 + ch] = fill_color[ch];
											}
										}
									}
								}
								else
								{
									NOT_IMPLEMENTED;
								}

								delete[] fill_color;
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
							Dtype fill_color = Dtype(color.g / 3 + color.b / 3 + color.r / 3);

							for (int n = 0; n < num; n++)
							{
								int n_offset = n * num_offset;
								Dtype *tptr0 = dst_data + n_offset + y11 * width;
								Dtype *tptr1 = dst_data + n_offset + y12 * width;
								for (int i = x11; i < x12; i++)
								{
									tptr0[i] = fill_color;
									tptr1[i] = fill_color;
								}

								tptr0 = dst_data + n_offset + y21 * width;
								tptr1 = dst_data + n_offset + y22 * width;
								for (int i = x21; i < x22; i++)
								{
									tptr0[i] = fill_color;
									tptr1[i] = fill_color;
								}
							}
						}
						else if (channels == 3 || channels == 4)
						{
							Dtype* fill_color = new Dtype[3];
							fill_color[0] = color.r;
							fill_color[1] = color.g;
							fill_color[2] = color.b;

							if (dst->order() == memory::NCHW)
							{
								for (int n = 0; n < num; n++)
								{
									int n_offset = n * num_offset;
									for (int ch = 0; ch < 3; ++ch)
									{
										int channel_offset = ch * offset;

										Dtype *tptr0 = dst_data + n_offset + channel_offset + y11 * width;
										Dtype *tptr1 = dst_data + n_offset + channel_offset + y12 * width;
										for (int i = x11; i < x12; i++)
										{
											tptr0[i] = fill_color[ch];
											tptr1[i] = fill_color[ch];
										}


										tptr0 = dst_data + n_offset + channel_offset + y21 * width;
										tptr1 = dst_data + n_offset + channel_offset + y22 * width;
										for (int i = x21; i < x22; i++)
										{
											tptr0[i] = fill_color[ch];
											tptr1[i] = fill_color[ch];
										}
									}
								}
							}
							else if (dst->order() == memory::NHWC)
							{
								int pos_x11_y11 = (y11 * width + x11) * 3;
								int pos_x11_y12 = (y12 * width + x11) * 3;
								for (int n = 0; n < num; n++)
								{
									int n_offset = n * num_offset;
									for (int col = 0; col < x12 - x11; ++col)
									{
										for (int ch = 0; ch < 3; ++ch)
										{
											dst_data[n_offset + pos_x11_y11 + col * 3 + ch] = fill_color[ch];
											dst_data[n_offset + pos_x11_y12 + col * 3 + ch] = fill_color[ch];
										}
									}
								}

								int pos_x21_y21 = (y21 * width + x21) * 3;
								int pos_x21_y22 = (y22 * width + x21) * 3;
								for (int n = 0; n < num; n++)
								{
									int n_offset = n * num_offset;
									for (int col = 0; col < x22 - x21; ++col)
									{
										for (int ch = 0; ch < 3; ++ch)
										{
											dst_data[n_offset + pos_x21_y21 + col * 3 + ch] = fill_color[ch];
											dst_data[n_offset + pos_x21_y22 + col * 3 + ch] = fill_color[ch];
										}
									}
								}
							}
							else
							{
								NOT_IMPLEMENTED;
							}

							delete[] fill_color;
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
			/// draw circle
			/// </summary>
			/// <param name="dst">memory::tensor with image data</param>
			/// <param name="center">circle center</param>
			/// <param name="radius">radius</param>
			/// <param name="color">color of lines</param>
			/// <param name="thickness">thickness of lines, 2 pixels by default</param>
			template <typename Dtype, typename Rtype>
			static void draw_circle_cpu(memory::tensor<Dtype>& dst, point<Rtype> center, int radius, color color, int thickness = 2)
			{
				if (dst.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = dst.num();
				int channels = dst.channels();
				int height = dst.height();
				int width = dst.width();
				int offset = height * width;
				int num_offset = channels * height * width;
				Dtype* dst_data = dst.mutable_cpu_data();

				if (center.x < radius || center.x >= width - radius || center.y < radius || center.y >= height - radius)
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
								Dtype fill_color = Dtype(color.g / 3 + color.b / 3 + color.r / 3);

								for (int n = 0; n < num; n++)
								{
									int n_offset = n * num_offset;
									Dtype *tptr0 = dst_data + n_offset + y11 * width;
									Dtype *tptr1 = dst_data + n_offset + y12 * width;
									for (int i = x11; i < x11 + 2; i++)
									{
										tptr0[i] = fill_color;
										tptr1[i] = fill_color;
									}

									for (int i = x12; i < x12 + 2; i++)
									{
										tptr0[i] = fill_color;
										tptr1[i] = fill_color;
									}

									tptr0 = dst_data + n_offset + y21 * width;
									tptr1 = dst_data + n_offset + y22 * width;
									for (int i = x21; i < x21 + 2; i++)
									{
										tptr0[i] = fill_color;
										tptr1[i] = fill_color;
									}

									for (int i = x22; i < x22 + 2; i++)
									{
										tptr0[i] = fill_color;
										tptr1[i] = fill_color;
									}
								}
							}
							else if (channels == 3 || channels == 4)
							{
								Dtype* fill_color = new Dtype[3];
								fill_color[0] = color.r;
								fill_color[1] = color.g;
								fill_color[2] = color.b;

								if (dst.order() == memory::NCHW)
								{
									for (int n = 0; n < num; n++)
									{
										int n_offset = n * num_offset;
										for (int ch = 0; ch < 3; ++ch)
										{
											int channel_offset = ch * offset;

											Dtype *tptr0 = dst_data + n_offset + channel_offset + y11 * width;
											Dtype *tptr1 = dst_data + n_offset + channel_offset + y12 * width;
											for (int i = x11; i < x11 + 2; i++)
											{
												tptr0[i] = fill_color[ch];
												tptr1[i] = fill_color[ch];
											}

											for (int i = x12; i < x12 + 2; i++)
											{
												tptr0[i] = fill_color[ch];
												tptr1[i] = fill_color[ch];
											}

											tptr0 = dst_data + n_offset + channel_offset + y21 * width;
											tptr1 = dst_data + n_offset + channel_offset + y22 * width;
											for (int i = x21; i < x21 + 2; i++)
											{
												tptr0[i] = fill_color[ch];
												tptr1[i] = fill_color[ch];
											}

											for (int i = x22; i < x22 + 2; i++)
											{
												tptr0[i] = fill_color[ch];
												tptr1[i] = fill_color[ch];
											}
										}
									}
								}
								else if (dst.order() == memory::NHWC)
								{
									int pos_x11_y11 = (y11 * width + x11) * 3;
									int pos_x11_y12 = (y12 * width + x11) * 3;
									int pos_x12_y11 = (y11 * width + x12) * 3;
									int pos_x12_y12 = (y12 * width + x12) * 3;

									int pos_x21_y21 = (y21 * width + x21) * 3;
									int pos_x21_y22 = (y22 * width + x21) * 3;
									int pos_x22_y21 = (y21 * width + x22) * 3;
									int pos_x22_y22 = (y22 * width + x22) * 3;

									for (int n = 0; n < num; n++)
									{
										int n_offset = n * num_offset;
										for (int col = 0; col < 2; ++col)//fill neighboring 2 pixels
										{
											for (int ch = 0; ch < 3; ++ch)
											{
												dst_data[n_offset + pos_x11_y11 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x11_y12 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x12_y11 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x12_y12 + col * 3 + ch] = fill_color[ch];
														 
												dst_data[n_offset + pos_x21_y21 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x21_y22 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x22_y21 + col * 3 + ch] = fill_color[ch];
												dst_data[n_offset + pos_x22_y22 + col * 3 + ch] = fill_color[ch];
											}
										}
									}
								}
								else
								{
									NOT_IMPLEMENTED;
								}

								delete[] fill_color;
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
							Dtype fill_color = Dtype(color.g / 3 + color.b / 3 + color.r / 3);

							for (int n = 0; n < num; n++)
							{
								int n_offset = n * num_offset;
								Dtype *tptr0 = dst_data + n_offset + y11 * width;
								Dtype *tptr1 = dst_data + n_offset + y12 * width;
								for (int i = x11; i < x12; i++)
								{
									tptr0[i] = fill_color;
									tptr1[i] = fill_color;
								}

								tptr0 = dst_data + n_offset + y21 * width;
								tptr1 = dst_data + n_offset + y22 * width;
								for (int i = x21; i < x22; i++)
								{
									tptr0[i] = fill_color;
									tptr1[i] = fill_color;
								}
							}
						}
						else if (channels == 3 || channels == 4)
						{
							Dtype* fill_color = new Dtype[3];
							fill_color[0] = color.r;
							fill_color[1] = color.g;
							fill_color[2] = color.b;

							if (dst.order() == memory::NCHW)
							{
								for (int n = 0; n < num; n++)
								{
									int n_offset = n * num_offset;
									for (int ch = 0; ch < 3; ++ch)
									{
										int channel_offset = ch * offset;

										Dtype *tptr0 = dst_data + n_offset + channel_offset + y11 * width;
										Dtype *tptr1 = dst_data + n_offset + channel_offset + y12 * width;
										for (int i = x11; i < x12; i++)
										{
											tptr0[i] = fill_color[ch];
											tptr1[i] = fill_color[ch];
										}

										tptr0 = dst_data + n_offset + channel_offset + y21 * width;
										tptr1 = dst_data + n_offset + channel_offset + y22 * width;
										for (int i = x21; i < x22; i++)
										{
											tptr0[i] = fill_color[ch];
											tptr1[i] = fill_color[ch];
										}
									}
								}
							}
							else if (dst.order() == memory::NHWC)
							{
								int pos_x11_y11 = (y11 * width + x11) * 3;
								int pos_x11_y12 = (y12 * width + x11) * 3;
								for (int n = 0; n < num; n++)
								{
									int n_offset = n * num_offset;
									for (int col = 0; col < x12 - x11; ++col)
									{
										for (int ch = 0; ch < 3; ++ch)
										{
											dst_data[n_offset + pos_x11_y11 + col * 3 + ch] = fill_color[ch];
											dst_data[n_offset + pos_x11_y12 + col * 3 + ch] = fill_color[ch];
										}
									}
								}

								int pos_x21_y21 = (y21 * width + x21) * 3;
								int pos_x21_y22 = (y22 * width + x21) * 3;
								for (int n = 0; n < num; n++)
								{
									int n_offset = n * num_offset;
									for (int col = 0; col < x22 - x21; ++col)
									{
										for (int ch = 0; ch < 3; ++ch)
										{
											dst_data[n_offset + pos_x21_y21 + col * 3 + ch] = fill_color[ch];
											dst_data[n_offset + pos_x21_y22 + col * 3 + ch] = fill_color[ch];
										}
									}
								}
							}
							else
							{
								NOT_IMPLEMENTED;
							}

							delete[] fill_color;
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
			/// flip image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="axis">flip axis: Width_Wise(default) / Height_Wise / Center_Wise(flip both height and width) / Channel_Wise</param>
			template <typename Dtype>
			static void flip_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst, flipType axis = Width_Wise)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst_temp->mutable_cpu_data();

				if (src->order() == memory::NCHW)
				{
					if (axis == Width_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

								for (int row = 0; row < height; ++row)
								{
									int index = channel_offset + row * width;
									for (int col = 0; col < width; ++col)
									{
										dst_data[n_offset + index + col] = src_data[n_offset + index + (width - col - 1)];
									}
								}
							}
						}
					}
					else if (axis == Height_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

								for (int row = 0; row < height; ++row)
								{
									int dst_index = channel_offset + row * width;
									int src_index = channel_offset + (height - row - 1) * width;
									memcpy(dst_data + n_offset + dst_index, src_data + n_offset + src_index, width * sizeof(Dtype));
								}
							}
						}
					}
					else if (axis == Center_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

								for (int row = 0; row < height; ++row)
								{
									int dst_index = channel_offset + row * width;
									int src_index = channel_offset + (height - row - 1) * width;
									for (int col = 0; col < width; ++col)
									{
										dst_data[n_offset + dst_index + col] = src_data[n_offset + src_index + (width - col - 1)];
									}
								}
							}
						}
					}
					else if (axis == Channel_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int dst_channel_offset = ch * offset;
								int src_channel_offset = (channels - ch - 1) * offset;

								for (int row = 0; row < height; ++row)
								{
									int row_offset = row * width;
									int dst_index = dst_channel_offset + row_offset;
									int src_index = src_channel_offset + row_offset;
									memcpy(dst_data + n_offset + dst_index, src_data + n_offset + src_index, width * sizeof(Dtype));
								}
							}
						}
					}
					else
					{
						LOG(ERROR) << "Un-support flip type.";
					}
				}
				else if (src->order() == memory::NHWC)
				{
					if (axis == Width_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int row = 0; row < height; ++row)
							{
								int row_pos = row * width * channels;
								for (int col = 0; col < width; ++col)
								{
									int dst_pos = row_pos + col * channels;
									int src_pos = row_pos + (width - col - 1) * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[n_offset + dst_pos + ch] = src_data[n_offset + src_pos + ch];
									}
								}
							}
						}
					}
					else if (axis == Height_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
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
										dst_data[n_offset + dst_pos2 + ch] = src_data[n_offset + src_pos2 + ch];
									}
								}
							}
						}
					}
					else if (axis == Center_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
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
										dst_data[n_offset + dst_pos2 + ch] = src_data[n_offset + src_pos2 + ch];
									}
								}
							}
						}
					}
					else if (axis == Channel_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int row = 0; row < height; ++row)
							{
								int row_pos = row * width * channels;
								for (int col = 0; col < width; ++col)
								{
									int col_pos = row_pos + col * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[n_offset + col_pos + ch] = src_data[n_offset + col_pos + (channels - ch - 1)];
									}
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

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// flip image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="axis">flip axis: Width_Wise(default) / Height_Wise / Center_Wise(flip both height and width) / Channel_Wise</param>
			template <typename Dtype>
			static void flip_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype>& dst, flipType axis = Width_Wise)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;

				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());
				const Dtype* src_data = src.cpu_data();
				Dtype* dst_data = dst_temp.mutable_cpu_data();

				if (src.order() == memory::NCHW)
				{
					if (axis == Width_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

								for (int row = 0; row < height; ++row)
								{
									int index = channel_offset + row * width;
									for (int col = 0; col < width; ++col)
									{
										dst_data[n_offset + index + col] = src_data[n_offset + index + (width - col - 1)];
									}
								}
							}
						}
					}
					else if (axis == Height_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

								for (int row = 0; row < height; ++row)
								{
									int dst_index = channel_offset + row * width;
									int src_index = channel_offset + (height - row - 1) * width;
									memcpy(dst_data + n_offset + dst_index, src_data + n_offset + src_index, width * sizeof(Dtype));
								}
							}
						}
					}
					else if (axis == Center_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

								for (int row = 0; row < height; ++row)
								{
									int dst_index = channel_offset + row * width;
									int src_index = channel_offset + (height - row - 1) * width;
									for (int col = 0; col < width; ++col)
									{
										dst_data[n_offset + dst_index + col] = src_data[n_offset + src_index + (width - col - 1)];
									}
								}
							}
						}
					}
					else if (axis == Channel_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int dst_channel_offset = ch * offset;
								int src_channel_offset = (channels - ch - 1) * offset;

								for (int row = 0; row < height; ++row)
								{
									int row_offset = row * width;
									int dst_index = dst_channel_offset + row_offset;
									int src_index = src_channel_offset + row_offset;
									memcpy(dst_data + n_offset + dst_index, src_data + n_offset + src_index, width * sizeof(Dtype));
								}
							}
						}
					}
					else
					{
						LOG(ERROR) << "Un-support flip type.";
					}
				}
				else if (src.order() == memory::NHWC)
				{
					if (axis == Width_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int row = 0; row < height; ++row)
							{
								int row_pos = row * width * channels;
								for (int col = 0; col < width; ++col)
								{
									int dst_pos = row_pos + col * channels;
									int src_pos = row_pos + (width - col - 1) * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[n_offset + dst_pos + ch] = src_data[n_offset + src_pos + ch];
									}
								}
							}
						}
					}
					else if (axis == Height_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
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
										dst_data[n_offset + dst_pos2 + ch] = src_data[n_offset + src_pos2 + ch];
									}
								}
							}
						}
					}
					else if (axis == Center_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
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
										dst_data[n_offset + dst_pos2 + ch] = src_data[n_offset + src_pos2 + ch];
									}
								}
							}
						}
					}
					else if (axis == Channel_Wise)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int row = 0; row < height; ++row)
							{
								int row_pos = row * width * channels;
								for (int col = 0; col < width; ++col)
								{
									int col_pos = row_pos + col * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[n_offset + col_pos + ch] = src_data[n_offset + col_pos + (channels - ch - 1)];
									}
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

				dst = dst_temp.clone();
			}



			/// <summary>
			/// convert 3 channels image to 1 channel
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			static void rgb2gray_cpu(const std::shared_ptr<memory::tensor<unsigned char>> &src, std::shared_ptr<memory::tensor<unsigned char>>& dst)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;

				if (!(channels == 3 || channels == 4))
				{
					LOG(ERROR) << "Incorrect input channel.";
					return;
				}

				std::shared_ptr<memory::tensor<unsigned char>> dst_temp;

				if (src->order() == memory::NCHW)
				{
					dst_temp.reset(new memory::tensor<unsigned char>(std::vector<int>{num, 1, height, width}, src->device(), src->order()));
					const unsigned char* src_data = src->cpu_data();
					unsigned char* dst_data = dst_temp->mutable_cpu_data();

#if SIMD_TYPE >= SIMDTYPE_SSE

					//B / G / R
					mm_type factor_float32[] = { mm_set1_ps(0.114f), mm_set1_ps(0.587f), mm_set1_ps(0.299f) };
					std::shared_ptr<memory::tensor<float>> temp_float_tensor = std::make_shared<memory::tensor<float>>(mm_align_size);
					float *temp_float_data = temp_float_tensor->mutable_cpu_data();

					__m128i temp_uint8_B, temp_uint8_G, temp_uint8_R;
					mm_type temp_float32_B, temp_float32_G, temp_float32_R, temp_float32_sum;
					mm_typei temp_int32_B, temp_int32_G, temp_int32_R;

					int circle_num = offset / mm_align_size;
					int index = 0;

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						index = 0;

						for (; index < circle_num; index++)
						{
							temp_uint8_B = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + index * mm_align_size));
							temp_uint8_G = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + offset + index * mm_align_size));
							temp_uint8_R = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + 2 * offset + index * mm_align_size));
							temp_int32_B = mm_cvtepu8_epi32(temp_uint8_B);
							temp_int32_G = mm_cvtepu8_epi32(temp_uint8_G);
							temp_int32_R = mm_cvtepu8_epi32(temp_uint8_R);
							temp_float32_B = mm_cvtepi32_ps(temp_int32_B);
							temp_float32_G = mm_cvtepi32_ps(temp_int32_G);
							temp_float32_R = mm_cvtepi32_ps(temp_int32_R);
							temp_float32_B = mm_mul_ps(temp_float32_B, factor_float32[0]);
							temp_float32_G = mm_mul_ps(temp_float32_G, factor_float32[1]);
							temp_float32_R = mm_mul_ps(temp_float32_R, factor_float32[2]);
							temp_float32_sum = mm_add_ps(temp_float32_B, temp_float32_G);
							temp_float32_sum = mm_add_ps(temp_float32_sum, temp_float32_R);
							mm_store_ps(temp_float_data, temp_float32_sum);
							for (int i = 0; i < mm_align_size; i++)
							{
								dst_data[n * offset + index * mm_align_size + i] = (unsigned char)(temp_float_data[i]);
							}
						}

						for (index *= mm_align_size; index < offset; index++)
						{
							//pixel order in opencv: B / G / R
							//convert formula: gray = 0.114 * B + 0.587 * G + 0.299 * R
							dst_data[n * offset + index] = (unsigned char)(src_data[n_offset + index] * 0.114f +
																		   src_data[n_offset + offset * 1 + index] * 0.587f +
																		   src_data[n_offset + offset * 2 + index] * 0.299f);
						}
					}
#else
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int index = 0; index < offset; index++)
						{
							//pixel order in opencv: B / G / R
							//convert formula: gray = 0.114 * B + 0.587 * G + 0.299 * R
							dst_data[n * offset + index] = (unsigned char)(src_data[n_offset + index] * 0.114f +
																	       src_data[n_offset + offset * 1 + index] * 0.587f +
																		   src_data[n_offset + offset * 2 + index] * 0.299f);
						}
					}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

				}
				else if (src->order() == memory::NHWC)
				{
					dst_temp.reset(new memory::tensor<unsigned char>(std::vector<int>{num, height, width, 1}, src->device(), src->order()));
					const unsigned char* src_data = src->cpu_data();
					unsigned char* dst_data = dst_temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						for (int row = 0; row < height; ++row)
						{
							int dst_pos1 = row * width;
							int src_pos1 = row * width * channels;
							for (int col = 0; col < width; ++col)
							{
								int dst_pos2 = dst_pos1 + col;
								int src_pos2 = src_pos1 + col * channels;
								//pixel order in opencv: B / G / R
								//convert formula: gray = 0.114 * B + 0.587 * G + 0.299 * R
								dst_data[n * offset + dst_pos2] = (unsigned char)(src_data[n * num_offset + src_pos2] * 0.114f +
																				  src_data[n * num_offset + src_pos2 + 1] * 0.587f +
																				  src_data[n * num_offset + src_pos2 + 2] * 0.299f);
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<unsigned char>>(dst_temp->clone());
			}



			/// <summary>
			/// convert 3 channels image to 1 channel
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			static void rgb2gray_cpu(const memory::tensor<unsigned char> &src, memory::tensor<unsigned char>& dst)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;

				if (!(channels == 3 || channels == 4))
				{
					LOG(ERROR) << "Incorrect input channel.";
					return;
				}

				memory::tensor<unsigned char> dst_temp;

				if (src.order() == memory::NCHW)
				{
					dst_temp = memory::tensor<unsigned char>(std::vector<int>{num, 1, height, width}, src.device(), src.order());
					const unsigned char* src_data = src.cpu_data();
					unsigned char* dst_data = dst_temp.mutable_cpu_data();

#if SIMD_TYPE >= SIMDTYPE_SSE

					//B / G / R
					mm_type factor_float32[] = { mm_set1_ps(0.114f), mm_set1_ps(0.587f), mm_set1_ps(0.299f) };
					std::shared_ptr<memory::tensor<float>> temp_float_tensor = std::make_shared<memory::tensor<float>>(mm_align_size);
					float *temp_float_data = temp_float_tensor->mutable_cpu_data();

					__m128i temp_uint8_B, temp_uint8_G, temp_uint8_R;
					mm_type temp_float32_B, temp_float32_G, temp_float32_R, temp_float32_sum;
					mm_typei temp_int32_B, temp_int32_G, temp_int32_R;

					int circle_num = offset / mm_align_size;
					int index = 0;

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						index = 0;

						for (; index < circle_num; index++)
						{
							temp_uint8_B = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + index * mm_align_size));
							temp_uint8_G = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + offset + index * mm_align_size));
							temp_uint8_R = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + 2 * offset + index * mm_align_size));
							temp_int32_B = mm_cvtepu8_epi32(temp_uint8_B);
							temp_int32_G = mm_cvtepu8_epi32(temp_uint8_G);
							temp_int32_R = mm_cvtepu8_epi32(temp_uint8_R);
							temp_float32_B = mm_cvtepi32_ps(temp_int32_B);
							temp_float32_G = mm_cvtepi32_ps(temp_int32_G);
							temp_float32_R = mm_cvtepi32_ps(temp_int32_R);
							temp_float32_B = mm_mul_ps(temp_float32_B, factor_float32[0]);
							temp_float32_G = mm_mul_ps(temp_float32_G, factor_float32[1]);
							temp_float32_R = mm_mul_ps(temp_float32_R, factor_float32[2]);
							temp_float32_sum = mm_add_ps(temp_float32_B, temp_float32_G);
							temp_float32_sum = mm_add_ps(temp_float32_sum, temp_float32_R);
							mm_store_ps(temp_float_data, temp_float32_sum);
							for (int i = 0; i < mm_align_size; i++)
							{
								dst_data[n * offset + index * mm_align_size + i] = (unsigned char)(temp_float_data[i]);
							}
						}

						for (index *= mm_align_size; index < offset; index++)
						{
							//pixel order in opencv: B / G / R
							//convert formula: gray = 0.114 * B + 0.587 * G + 0.299 * R
							dst_data[n * offset + index] = (unsigned char)(src_data[n_offset + index] * 0.114f +
																		   src_data[n_offset + offset * 1 + index] * 0.587f +
																		   src_data[n_offset + offset * 2 + index] * 0.299f);
						}
					}
#else
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int index = 0; index < offset; index++)
						{
							//pixel order in opencv: B / G / R
							//convert formula: gray = 0.114 * B + 0.587 * G + 0.299 * R
							dst_data[n * offset + index] = (unsigned char)(src_data[n_offset + index] * 0.114f +
																		   src_data[n_offset + offset * 1 + index] * 0.587f +
																		   src_data[n_offset + offset * 2 + index] * 0.299f);
						}
					}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

				}
				else if (src.order() == memory::NHWC)
				{
					dst_temp = memory::tensor<unsigned char>(std::vector<int>{num, height, width, 1}, src.device(), src.order());
					const unsigned char* src_data = src.cpu_data();
					unsigned char* dst_data = dst_temp.mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						for (int row = 0; row < height; ++row)
						{
							int dst_pos1 = row * width;
							int src_pos1 = row * width * channels;
							for (int col = 0; col < width; ++col)
							{
								int dst_pos2 = dst_pos1 + col;
								int src_pos2 = src_pos1 + col * channels;
								//pixel order in opencv: B / G / R
								//convert formula: gray = 0.114 * B + 0.587 * G + 0.299 * R
								dst_data[n * offset + dst_pos2] = (unsigned char)(src_data[n * num_offset + src_pos2] * 0.114f +
																		          src_data[n * num_offset + src_pos2 + 1] * 0.587f +
																				  src_data[n * num_offset + src_pos2 + 2] * 0.299f);
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// convert rgb image to hsv image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			static void rgb2hsv_cpu(const std::shared_ptr<memory::tensor<unsigned char>> &src, std::shared_ptr<memory::tensor<unsigned char>>& dst)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;

				if (!(channels == 3 || channels == 4))
				{
					LOG(ERROR) << "Incorrect input channel.";
					return;
				}

				std::shared_ptr<memory::tensor<unsigned char>> dst_temp;
				dst_temp.reset(new memory::tensor<unsigned char>(std::vector<int>{num, 3, height, width}, src->device(), memory::NCHW));
				const unsigned char* src_data = src->cpu_data();
				unsigned char* dst_data = dst_temp->mutable_cpu_data();

				if (src->order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

						for (int row = 0; row < height; row++)
						{
							int row_offset = row * width;
							for (int col = 0; col < width; col++)
							{
								std::vector<unsigned char> pixel;
								unsigned char B = src_data[n_offset + row_offset + col];
								unsigned char G = src_data[n_offset + offset + row_offset + col];
								unsigned char R = src_data[n_offset + 2 * offset + row_offset + col];
								pixel.push_back(B);
								pixel.push_back(G);
								pixel.push_back(R);
								std::sort(pixel.begin(), pixel.end());

								//v
								dst_data[n_offset + 2 * offset + row_offset + col] = pixel[2];

								//s
								if (pixel[2] == 0)
								{
									dst_data[n_offset + offset + row_offset + col] = 0;
								}
								else
								{
									dst_data[n_offset + offset + row_offset + col] = 255 * (pixel[2] - pixel[0]) / pixel[2];
								}

								//h
								int H;
								if (pixel[2] == pixel[0])
								{
									H = 0;
								}
								else if ((pixel[2] == R) && (G >= B))
								{
									H = 60 * (G - B) / (pixel[2] - pixel[0]);
								}
								else if ((pixel[2] == R) && (G < B))
								{
									H = 60 * (G - B) / (pixel[2] - pixel[0]) + 360;
								}
								else if (pixel[2] == G)
								{
									H = 120 + 60 * (B - R) / (pixel[2] - pixel[0]);
								}
								else if (pixel[2] == B)
								{
									H = 240 + 60 * (R - G) / (pixel[2] - pixel[0]);
								}

								dst_data[n_offset + row * width + col] = H / 2;
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

						for (int row = 0; row < height; ++row)
						{
							int row_offset = channels * row * width;
							for (int col = 0; col < width; ++col)
							{
								int col_offset = channels * col;
								std::vector<unsigned char> pixel;
								unsigned char B = src_data[n_offset + row_offset + col_offset];
								unsigned char G = src_data[n_offset + row_offset + col_offset + 1];
								unsigned char R = src_data[n_offset + row_offset + col_offset + 2];
								pixel.push_back(B);
								pixel.push_back(G);
								pixel.push_back(R);
								std::sort(pixel.begin(), pixel.end());

								//v
								dst_data[n_offset + 2 * offset + row * width + col] = pixel[2];

								//s
								if (pixel[2] == 0)
								{
									dst_data[n_offset + offset + row * width + col] = 0;
								}
								else
								{
									dst_data[n_offset + offset + row * width + col] = 255 * (pixel[2] - pixel[0]) / pixel[2];
								}

								//h
								int H;
								if (pixel[2] == pixel[0])
								{
									H = 0;
								}
								else if ((pixel[2] == R) && (G >= B))
								{
									H = 60 * (G - B) / (pixel[2] - pixel[0]);
								}
								else if ((pixel[2] == R) && (G < B))
								{
									H = 60 * (G - B) / (pixel[2] - pixel[0]) + 360;
								}
								else if (pixel[2] == G)
								{
									H = 120 + 60 * (B - R) / (pixel[2] - pixel[0]);
								}
								else if (pixel[2] == B)
								{
									H = 240 + 60 * (R - G) / (pixel[2] - pixel[0]);
								}

								dst_data[n_offset + row * width + col] = H / 2;
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<unsigned char>>(dst_temp->clone());
			}



			/// <summary>
			/// convert rgb image to hsv image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			static void rgb2hsv_cpu(const memory::tensor<unsigned char> &src, memory::tensor<unsigned char>& dst)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;

				if (!(channels == 3 || channels == 4))
				{
					LOG(ERROR) << "Incorrect input channel.";
					return;
				}

				memory::tensor<unsigned char> dst_temp = memory::tensor<unsigned char>(std::vector<int>{num, 3, height, width}, src.device(), memory::NCHW);
				const unsigned char* src_data = src.cpu_data();
				unsigned char* dst_data = dst_temp.mutable_cpu_data();

				if (src.order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

						for (int row = 0; row < height; row++)
						{
							int row_offset = row * width;
							for (int col = 0; col < width; col++)
							{
								std::vector<unsigned char> pixel;
								unsigned char B = src_data[n_offset + row_offset + col];
								unsigned char G = src_data[n_offset + offset + row_offset + col];
								unsigned char R = src_data[n_offset + 2 * offset + row_offset + col];
								pixel.push_back(B);
								pixel.push_back(G);
								pixel.push_back(R);
								std::sort(pixel.begin(), pixel.end());

								//v
								dst_data[n_offset + 2 * offset + row_offset + col] = pixel[2];

								//s
								if (pixel[2] == 0)
								{
									dst_data[n_offset + offset + row_offset + col] = 0;
								}
								else
								{
									dst_data[n_offset + offset + row_offset + col] = 255 * (pixel[2] - pixel[0]) / pixel[2];
								}

								//h
								int H;
								if (pixel[2] == pixel[0])
								{
									H = 0;
								}
								else if ((pixel[2] == R) && (G >= B))
								{
									H = 60 * (G - B) / (pixel[2] - pixel[0]);
								}
								else if ((pixel[2] == R) && (G < B))
								{
									H = 60 * (G - B) / (pixel[2] - pixel[0]) + 360;
								}
								else if (pixel[2] == G)
								{
									H = 120 + 60 * (B - R) / (pixel[2] - pixel[0]);
								}
								else if (pixel[2] == B)
								{
									H = 240 + 60 * (R - G) / (pixel[2] - pixel[0]);
								}

								dst_data[n_offset + row * width + col] = H / 2;
							}
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

						for (int row = 0; row < height; ++row)
						{
							int row_offset = channels * row * width;
							for (int col = 0; col < width; ++col)
							{
								int col_offset = channels * col;
								std::vector<unsigned char> pixel;
								unsigned char B = src_data[n_offset + row_offset + col_offset];
								unsigned char G = src_data[n_offset + row_offset + col_offset + 1];
								unsigned char R = src_data[n_offset + row_offset + col_offset + 2];
								pixel.push_back(B);
								pixel.push_back(G);
								pixel.push_back(R);
								std::sort(pixel.begin(), pixel.end());

								//v
								dst_data[n_offset + 2 * offset + row * width + col] = pixel[2];

								//s
								if (pixel[2] == 0)
								{
									dst_data[n_offset + offset + row * width + col] = 0;
								}
								else
								{
									dst_data[n_offset + offset + row * width + col] = 255 * (pixel[2] - pixel[0]) / pixel[2];
								}

								//h
								int H;
								if (pixel[2] == pixel[0])
								{
									H = 0;
								}
								else if ((pixel[2] == R) && (G >= B))
								{
									H = 60 * (G - B) / (pixel[2] - pixel[0]);
								}
								else if ((pixel[2] == R) && (G < B))
								{
									H = 60 * (G - B) / (pixel[2] - pixel[0]) + 360;
								}
								else if (pixel[2] == G)
								{
									H = 120 + 60 * (B - R) / (pixel[2] - pixel[0]);
								}
								else if (pixel[2] == B)
								{
									H = 240 + 60 * (R - G) / (pixel[2] - pixel[0]);
								}

								dst_data[n_offset + row * width + col] = H / 2;
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// matrix transpose
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename Dtype>
			static void matrix_transpose_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				if (src->order() == memory::NCHW)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, width, height}, src->device(), src->order()));
					const Dtype* src_data = src->cpu_data();
					Dtype* dst_data = dst_temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int ch = 0; ch < channels; ++ch) {
							int channel_offset = ch * offset;
							for (int row = 0; row < width; ++row)
							{
								int dst_index = channel_offset + row * height;
								for (int col = 0; col < height; ++col)
								{
									dst_data[n_offset + dst_index + col] = src_data[n_offset + channel_offset + col * width + row];
								}
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, width, height, channels}, src->device(), src->order()));
					const Dtype* src_data = src->cpu_data();
					Dtype* dst_data = dst_temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int row = 0; row < width; ++row)
						{
							int dst_pos1 = row * height * channels;;
							for (int col = 0; col < height; ++col)
							{
								int dst_pos2 = dst_pos1 + col * channels;
								int src_pos = (col * width + row) * channels;

								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[n_offset + dst_pos2 + ch] = src_data[n_offset + src_pos + ch];
								}
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// matrix transpose
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename Dtype>
			static void matrix_transpose_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype>& dst)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;

				memory::tensor<Dtype> dst_temp;
				if (src.order() == memory::NCHW)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, channels, width, height}, src.device(), src.order());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst_temp.mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int ch = 0; ch < channels; ++ch) {
							int channel_offset = ch * offset;
							for (int row = 0; row < width; ++row)
							{
								int dst_index = channel_offset + row * height;
								for (int col = 0; col < height; ++col)
								{
									dst_data[n_offset + dst_index + col] = src_data[n_offset + channel_offset + col * width + row];
								}
							}
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, width, height, channels}, src.device(), src.order());
					const Dtype* src_data = src.cpu_data();
					Dtype* dst_data = dst_temp.mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int row = 0; row < width; ++row)
						{
							int dst_pos1 = row * height * channels;
							for (int col = 0; col < height; ++col)
							{
								int dst_pos2 = dst_pos1 + col * channels;
								int src_pos = (col * width + row) * channels;

								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[n_offset + dst_pos2 + ch] = src_data[n_offset + src_pos + ch];
								}
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// get ROI(region of interest) from image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">ROI memory::tensor</param>
			/// <param name="rect">ROI rectangle</param>
			template <typename Dtype, typename Rtype>
			static void roi_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst, rectangle<Rtype> rect)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int src_offset = height * width;
				int src_num_offset = channels * height * width;

				if (rect.x < 0 || rect.x + rect.w >= width || rect.y < 0 || rect.y + rect.h >= height) {
					LOG(WARNING) << "rect is out of image";
					return;
				}

				int dst_height = (int)rect.h;
				int dst_width = (int)rect.w;
				int dst_offset = dst_height * dst_width;
				int dst_num_offset = channels * dst_height * dst_width;

				if (dst_height == height && dst_width == width)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				if (src->order() == memory::NCHW)
				{
					dst.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
					Dtype* dst_data = dst->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							for (int row = 0; row < dst_height; ++row)
							{
								int src_index = src_channel_offset + (row + rect.y) * width + rect.x;
								int dst_index = dst_channel_offset + row * dst_width;

								memcpy(dst_data + dst_n_offset + dst_index, src_data + src_n_offset + src_index, dst_width * sizeof(Dtype));
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					dst.reset(new memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
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
									dst_data[dst_n_offset + dst_pos2 + ch] = src_data[src_n_offset + src_pos2 + ch];
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
			/// get ROI(region of interest) from image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">ROI memory::tensor</param>
			/// <param name="rect">ROI rectangle</param>
			template <typename Dtype, typename Rtype>
			static void roi_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype>& dst, rectangle<Rtype> rect)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;
				int src_num_offset = channels * height * width;

				if (rect.x < 0 || rect.x + rect.w >= width || rect.y < 0 || rect.y + rect.h >= height)
				{
					LOG(WARNING) << "rect is out of image";
					return;
				}

				int dst_height = (int)rect.h;
				int dst_width = (int)rect.w;
				int dst_offset = dst_height * dst_width;
				int dst_num_offset = channels * dst_height * dst_width;

				if (dst_height == height && dst_width == width)
				{
					dst = src.clone();
					return;
				}

				if (src.order() == memory::NCHW)
				{
					dst = memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src.device(), src.order());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							for (int row = 0; row < dst_height; ++row)
							{
								int src_index = src_channel_offset + (row + rect.y) * width + rect.x;
								int dst_index = dst_channel_offset + row * dst_width;

								memcpy(dst_data + dst_n_offset + dst_index, src_data + src_n_offset + src_index, dst_width * sizeof(Dtype));
							}
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					dst = memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src.device(), src.order());
					Dtype* dst_data = dst.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
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
									dst_data[dst_n_offset + dst_pos2 + ch] = src_data[src_n_offset + src_pos2 + ch];
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
			/// expand border
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="top">pixels to expand at top of image</param>
			/// <param name="bottom">pixels to expand at bottom of image</param>
			/// <param name="left">pixels to expand at left of image</param>
			/// <param name="right">pixels to expand at right of image</param>
			/// <param name="type">borderType: Border_Constant(default, use constant pixel value(fill_pixel_value) to fill in new blank area) / Border_Replicate(replicate neighboring pixel to fill in new blank area)</param>
			/// <param name="fill_pixel_value">validate when borderType is Border_Constant, zero by default</param>
			template <typename Dtype>
			static void make_border_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst,
				int top, int bottom, int left, int right, borderType type = Border_Constant, Dtype fill_pixel_value = 0)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int src_offset = height * width;
				int src_num_offset = channels * height * width;

				int dst_height = height + top + bottom;
				int dst_width = width + left + right;
				int dst_offset = dst_height * dst_width;
				int dst_num_offset = channels * dst_height * dst_width;

				if (top < 0 || bottom < 0 || left < 0 || right < 0)
				{
					LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
					return;
				}

				if (dst_height == height && dst_width == width)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				if (src->order() == memory::NCHW)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
					Dtype* dst_data = dst_temp->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					if (type == Border_Constant)
					{
						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;
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
										dst_data[dst_n_offset + dst_index + col] = fill_pixel_value;
									}
								}

								//center
								for (int row = top; row < top + height; ++row)
								{
									int src_index = src_channel_offset + (row - top) * width;
									int dst_index = dst_channel_offset + row * dst_width;

									for (int col = 0; col < left; col++)
									{
										dst_data[dst_n_offset + dst_index + col] = fill_pixel_value;
									}

									memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));

									for (int col = left + width; col < dst_width; col++)
									{
										dst_data[dst_n_offset + dst_index + col] = fill_pixel_value;
									}
								}

								//bottom
								for (int row = top + height; row < dst_height; row++)
								{
									int dst_index = dst_channel_offset + row * dst_width;
									for (int col = 0; col < dst_width; col++)
									{
										dst_data[dst_n_offset + dst_index + col] = fill_pixel_value;
									}
								}
							}
						}
					}
					else if (type == Border_Replicate)
					{
						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;
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
										dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_channel_offset];
									}

									//center
									memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_channel_offset, width * sizeof(Dtype));

									//right
									for (int col = left + width; col < dst_width; col++)
									{
										dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_channel_offset + width - 1];
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
										dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index];
									}

									//center
									memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));

									//right
									for (int col = left + width; col < dst_width; col++)
									{
										dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index + width - 1];
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
										dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index];
									}

									//center
									memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));

									//right
									for (int col = left + width; col < dst_width; col++)
									{
										dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index + width - 1];
									}
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
				else if (src->order() == memory::NHWC)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst_temp->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					if (type == Border_Constant)
					{
						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;

							//top
							for (int row = 0; row < top; row++)
							{
								int dst_index1 = row * dst_width * channels;
								for (int col = 0; col < dst_width; col++)
								{
									int dst_index2 = dst_index1 + col * channels;
									for (int ch = 0; ch < channels; ch++)
									{
										dst_data[dst_n_offset + dst_index2 + ch] = fill_pixel_value;
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
										dst_data[dst_n_offset + dst_index2 + ch] = fill_pixel_value;
									}
								}

								//center
								memcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index, width * channels * sizeof(Dtype));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									int dst_index2 = dst_index1 + col * channels;
									for (int ch = 0; ch < channels; ch++)
									{
										dst_data[dst_n_offset + dst_index2 + ch] = fill_pixel_value;
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
										dst_data[dst_n_offset + dst_index2 + ch] = fill_pixel_value;
									}
								}
							}
						}
					}
					else if (type == Border_Replicate)
					{
						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;

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
										dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + ch];
									}
								}

								//center
								memcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset, width * channels * sizeof(Dtype));

								//right
								for (int col = left + width; col < dst_width; ++col)
								{
									int dst_index2 = dst_index1 + col * channels;
									int src_index = (width - 1) * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index + ch];
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
										dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index1 + ch];
									}
								}

								//center
								memcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index1, width * channels * sizeof(Dtype));

								//right
								int src_index2 = src_index1 + (width - 1) * channels;
								for (int col = left + width; col < dst_width; ++col)
								{
									int dst_index2 = dst_index1 + col * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index2 + ch];
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
										dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index1 + ch];
									}
								}

								//center
								memcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index1, width * channels * sizeof(Dtype));

								//right
								int src_index2 = src_index1 + (width - 1) * channels;
								for (int col = left + width; col < dst_width; ++col)
								{
									int dst_index2 = dst_index1 + col * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index2 + ch];
									}
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

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// expand border
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="top">pixels to expand at top of image</param>
			/// <param name="bottom">pixels to expand at bottom of image</param>
			/// <param name="left">pixels to expand at left of image</param>
			/// <param name="right">pixels to expand at right of image</param>
			/// <param name="type">borderType: Border_Constant(default, use constant pixel value(fill_pixel_value) to fill in new blank area) / Border_Replicate(replicate neighboring pixel to fill in new blank area)</param>
			/// <param name="fill_pixel_value">validate when borderType is Border_Constant, zero by default</param>
			template <typename Dtype>
			static void make_border_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype>& dst,
				int top, int bottom, int left, int right, borderType type = Border_Constant, Dtype fill_pixel_value = 0)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;
				int src_num_offset = channels * height * width;

				int dst_height = height + top + bottom;
				int dst_width = width + left + right;
				int dst_offset = dst_height * dst_width;
				int dst_num_offset = channels * dst_height * dst_width;

				if (top < 0 || bottom < 0 || left < 0 || right < 0)
				{
					LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
					return;
				}

				if (dst_height == height && dst_width == width)
				{
					dst = src.clone();
					return;
				}

				memory::tensor<Dtype> dst_temp;
				if (src.order() == memory::NCHW)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src.device(), src.order());
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					if (type == Border_Constant)
					{
						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int src_channel_offset = ch * src_offset;
								int dst_channel_offset = ch * dst_offset;

								//top
								for (int i = dst_channel_offset; i < dst_channel_offset + top * dst_width; i++)
								{
									dst_data[dst_n_offset + i] = fill_pixel_value;
								}

								//center
								for (int row = top; row < top + height; ++row)
								{
									int src_index = src_channel_offset + (row - top) * width;
									int dst_index = dst_channel_offset + row * dst_width;

									for (int i = dst_index; i < dst_index + left; i++)
									{
										dst_data[dst_n_offset + i] = fill_pixel_value;
									}

									memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));
									
									for (int i = dst_index + left + width; i < dst_index + left + width + right; i++)
									{
										dst_data[dst_n_offset + i] = fill_pixel_value;
									}
								}

								//bottom
								for (int i = dst_channel_offset + (top + height) * dst_width; i < dst_channel_offset + (top + height) * dst_width + bottom * dst_width; i++)
								{
									dst_data[dst_n_offset + i] = fill_pixel_value;
								}
							}
						}
					}
					else if (type == Border_Replicate)
					{
						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int src_channel_offset = ch * src_offset;
								int dst_channel_offset = ch * dst_offset;

								//top
								for (int row = 0; row < top; ++row)
								{
									int dst_index = dst_channel_offset + row * dst_width;
									for (int i = dst_index; i < dst_index + left; i++)
									{
										dst_data[dst_n_offset + i] = src_data[src_n_offset + src_channel_offset];
									}

									memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_channel_offset, width * sizeof(Dtype));
									
									for (int i = dst_index + left + width; i < dst_index + left + width + right; i++)
									{
										dst_data[dst_n_offset + i] = src_data[src_n_offset + src_channel_offset + width - 1];
									}
								}

								//center
								for (int row = top; row < top + height; ++row)
								{
									int src_index = src_channel_offset + (row - top) * width;
									int dst_index = dst_channel_offset + row * dst_width;

									for (int i = dst_index; i < dst_index + left; i++)
									{
										dst_data[dst_n_offset + i] = src_data[src_n_offset + src_index];
									}

									memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));
									
									for (int i = dst_index + left + width; i < dst_index + left + width + right; i++)
									{
										dst_data[dst_n_offset + i] = src_data[src_n_offset + src_index + width - 1];
									}
								}

								//bottom
								for (int row = top + height; row < dst_height; ++row)
								{
									int src_index = src_channel_offset + (height - 1) * width;
									int dst_index = dst_channel_offset + row * dst_width;

									for (int i = dst_index; i < dst_index + left; i++)
									{
										dst_data[dst_n_offset + i] = src_data[src_n_offset + src_index];
									}

									memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));
									
									for (int i = dst_index + left + width; i < dst_index + left + width + right; i++)
									{
										dst_data[dst_n_offset + i] = src_data[src_n_offset + src_index + width - 1];
									}
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
				else if (src.order() == memory::NHWC)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src.device(), src.order());
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					if (type == Border_Constant)
					{
						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;

							//top
							for (int i = 0; i < top * dst_width * channels; i++)
							{
								dst_data[dst_n_offset + i] = fill_pixel_value;
							}

							//center
							for (int row = top; row < top + height; ++row)
							{
								int src_pos = (row - top) * width * channels;

								//left
								int dst_pos1 = row * dst_width * channels;
								for (int i = dst_pos1; i < dst_pos1 + left * channels; i++)
								{
									dst_data[dst_n_offset + i] = fill_pixel_value;
								}

								//center
								int dst_pos2 = dst_pos1 + left * channels;
								memcpy(dst_data + dst_n_offset + dst_pos2, src_data + src_n_offset + src_pos, width * channels * sizeof(Dtype));

								//right
								int dst_pos3 = dst_pos2 + width * channels;
								for (int i = dst_pos3; i < dst_pos3 + right * channels; i++)
								{
									dst_data[dst_n_offset + i] = fill_pixel_value;
								}
							}

							//bottom
							for (int i = (top + height) * dst_width * channels; i < (top + height) * dst_width * channels + bottom * dst_width * channels; i++)
							{
								dst_data[dst_n_offset + i] = fill_pixel_value;
							}
						}
					}
					else if (type == Border_Replicate)
					{
						for (int n = 0; n < num; n++)
						{
							int src_n_offset = n * src_num_offset;
							int dst_n_offset = n * dst_num_offset;

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
										dst_data[dst_n_offset + dst_index + ch] = src_data[src_n_offset + ch];
									}
								}

								//center
								int dst_pos2 = dst_pos1 + left * channels;
								memcpy(dst_data + dst_n_offset + dst_pos2, src_data + src_n_offset, width * channels * sizeof(Dtype));

								//right
								int dst_pos3 = dst_pos2 + width * channels;
								for (int col = 0; col < right; ++col)
								{
									int dst_index = dst_pos3 + col * channels;
									int src_index = (width - 1) * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[dst_n_offset + dst_index + ch] = src_data[src_n_offset + src_index + ch];
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
										dst_data[dst_n_offset + dst_index + ch] = src_data[src_n_offset + src_pos1 + ch];
									}
								}

								//center
								int dst_pos2 = dst_pos1 + left * channels;
								memcpy(dst_data + dst_n_offset + dst_pos2, src_data + src_n_offset + src_pos1, width * channels * sizeof(Dtype));

								//right
								int src_pos2 = src_pos1 + width * channels;
								int dst_pos3 = dst_pos2 + width * channels;
								for (int col = 0; col < right; ++col)
								{
									int dst_index = dst_pos3 + col * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[dst_n_offset + dst_index + ch] = src_data[src_n_offset + src_pos2 + ch];
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
										dst_data[dst_n_offset + dst_index + ch] = src_data[src_n_offset + src_pos1 + ch];
									}
								}

								//center
								int dst_pos2 = dst_pos1 + left * channels;
								memcpy(dst_data + dst_n_offset + dst_pos2, src_data + src_n_offset + src_pos1, width * channels * sizeof(Dtype));

								//right
								int dst_pos3 = dst_pos2 + width * channels;
								int src_pos2 = src_pos1 + width * channels;
								for (int col = 0; col < right; ++col)
								{
									int dst_index = dst_pos3 + col * channels;
									for (int ch = 0; ch < channels; ++ch)
									{
										dst_data[dst_n_offset + dst_index + ch] = src_data[src_n_offset + src_pos2 + ch];
									}
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

				dst = dst_temp.clone();
			}



			/// <summary>
			/// cut border
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="top">pixels to cut at top of image</param>
			/// <param name="bottom">pixels to cut at bottom of image</param>
			/// <param name="left">pixels to cut at left of image</param>
			/// <param name="right">pixels to cut at right of image</param>
			template <typename Dtype>
			static void cut_border_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src,
				std::shared_ptr<memory::tensor<Dtype>>& dst, int top, int bottom, int left, int right)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (top < 0 || bottom < 0 || left < 0 || right < 0)
				{
					LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int src_offset = height * width;
				int src_num_offset = channels * height * width;

				int dst_height = height - top - bottom;
				int dst_width = width - left - right;
				int dst_offset = dst_height * dst_width;
				int dst_num_offset = channels * dst_height * dst_width;
				if (dst_height == height && dst_width == width)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				if (dst_height <= 0 || dst_width <= 0)
				{
					LOG(ERROR) << "Illegal input size.";
					return;
				}

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				if (src->order() == memory::NCHW)
				{					
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
					Dtype* dst_data = dst_temp->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							for (int row = 0; row < dst_height; ++row)
							{
								int src_index = src_channel_offset + (row + top) * width + left;
								int dst_index = dst_channel_offset + row * dst_width;
								memcpy(dst_data + dst_n_offset + dst_index, src_data + src_n_offset + src_index, dst_width * sizeof(Dtype));
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
					Dtype* dst_data = dst_temp->mutable_cpu_data();
					const Dtype* src_data = src->cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						for (int row = 0; row < dst_height; ++row)
						{
							int src_index = ((row + top) * width + left) * channels;
							int dst_index = row * dst_width * channels;
							memcpy(dst_data + dst_n_offset + dst_index, src_data + src_n_offset + src_index, dst_width * channels * sizeof(Dtype));
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// cut border
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="top">pixels to cut at top of image</param>
			/// <param name="bottom">pixels to cut at bottom of image</param>
			/// <param name="left">pixels to cut at left of image</param>
			/// <param name="right">pixels to cut at right of image</param>
			template <typename Dtype>
			static void cut_border_cpu(const memory::tensor<Dtype> &src,
				memory::tensor<Dtype>& dst, int top, int bottom, int left, int right)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (top < 0 || bottom < 0 || left < 0 || right < 0)
				{
					LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;
				int src_num_offset = channels * height * width;

				int dst_height = height - top - bottom;
				int dst_width = width - left - right;
				int dst_offset = dst_height * dst_width;
				int dst_num_offset = channels * dst_height * dst_width;

				dst = src.clone();
				if (dst_height == height && dst_width == width)
				{
					return;
				}

				if (dst_height <= 0 || dst_width <= 0)
				{
					LOG(ERROR) << "Illegal input size.";
					return;
				}

				memory::tensor<Dtype> dst_temp;
				if (src.order() == memory::NCHW)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src.device(), src.order());
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							for (int row = 0; row < dst_height; ++row)
							{
								int src_index = src_channel_offset + (row + top) * width + left;
								int dst_index = dst_channel_offset + row * dst_width;
								memcpy(dst_data + dst_n_offset + dst_index, src_data + src_n_offset + src_index, dst_width * sizeof(Dtype));
							}
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src.device(), src.order());
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = src.cpu_data();

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						for (int row = 0; row < dst_height; ++row)
						{
							int src_index = ((row + top) * width + left) * channels;
							int dst_index = row * dst_width * channels;
							memcpy(dst_data + dst_n_offset + dst_index, src_data + src_n_offset + src_index, dst_width * channels * sizeof(Dtype));
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// for image(w*h), calculate integral mapping((w+1)*(h+1)), first row and first col are zeros in integral mapping
			/// </summary>
			/// <param name="src">image memory::tensor</param>
			/// <param name="dst">integral mapping memory::tensor</param>
			template <typename Stype, typename Dtype>
			static void fast_integral_cpu(const std::shared_ptr<memory::tensor<Stype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int src_num_offset = channels * height * width;
				int dst_num_offset = channels * (height + 1) * (width + 1);

				if (src->order() == memory::NCHW)
				{
					dst.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, height + 1, width + 1}, src->device(), src->order()));
					const Stype* src_data = src->cpu_data();
					Dtype* dst_data = dst->mutable_cpu_data();
					memset(dst_data, (Dtype)0, num * channels * (height + 1) * (width + 1) * sizeof(Dtype));

					// sum of each column
					Dtype *columnSum = new Dtype[width + 1];

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						memset(columnSum, Dtype(0), (width + 1) * sizeof(Dtype));

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_offset = ch * height * width;
							int dst_offset = ch * (height + 1) * (width + 1);

							// calculate integral of the first nonzero_row(second row) 
							unsigned fist_nonzero_row_index = dst_offset + (width + 1);
							for (int col = 1; col < width + 1; ++col)
							{
								columnSum[col] = (Dtype)src_data[src_n_offset + src_offset + col - 1];
								dst_data[dst_n_offset + fist_nonzero_row_index + col] = (Dtype)src_data[src_n_offset + src_offset + col - 1];
								dst_data[dst_n_offset + fist_nonzero_row_index + col] += dst_data[dst_n_offset + fist_nonzero_row_index + col - 1];
							}

							for (int row = 2; row < height + 1; ++row)
							{
								int src_row_offset = (row - 1) * width;
								int dst_row_offset = row * (width + 1);

								for (int col = 1; col < width + 1; ++col)
								{
									columnSum[col] += (Dtype)src_data[src_n_offset + src_offset + src_row_offset + col - 1];
									dst_data[dst_n_offset + dst_offset + dst_row_offset + col] = dst_data[dst_n_offset + dst_offset + dst_row_offset + col - 1] + columnSum[col];
								}
							}
						}
					}

					delete[] columnSum;
				}
				else if (src->order() == memory::NHWC)
				{
					dst.reset(new memory::tensor<Dtype>(std::vector<int>{num, height + 1, width + 1, channels}, src->device(), src->order()));
					const Stype* src_data = src->cpu_data();
					Dtype* dst_data = dst->mutable_cpu_data();
					memset(dst_data, (Dtype)0, num * channels * (height + 1) * (width + 1) * sizeof(Dtype));

					// sum of each column
					Dtype *columnSum = new Dtype[width + 1];

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						memset(columnSum, Dtype(0), (width + 1) * sizeof(Dtype));

						for (int ch = 0; ch < channels; ++ch)
						{
							// calculate integral of the first nonzero_row(second row) 
							unsigned fist_nonzero_row_index = (width + 1) * channels + ch;
							for (int col = 1; col < width + 1; ++col)
							{
								columnSum[col] = (Dtype)src_data[src_n_offset + (col - 1) * channels + ch];
								dst_data[dst_n_offset + fist_nonzero_row_index + col * channels] = (Dtype)src_data[src_n_offset + (col - 1) * channels + ch];
								dst_data[dst_n_offset + fist_nonzero_row_index + col * channels] += dst_data[dst_n_offset + fist_nonzero_row_index + (col - 1) * channels];
							}

							for (int row = 2; row < height + 1; ++row)
							{
								int src_row_offset = (row - 1) * width * channels + ch;
								int dst_row_offset = row * (width + 1) * channels + ch;

								for (int col = 1; col < width + 1; ++col)
								{
									columnSum[col] += (Dtype)src_data[src_n_offset + src_row_offset + (col - 1) * channels];
									dst_data[dst_n_offset + dst_row_offset + col * channels] = dst_data[dst_n_offset + dst_row_offset + (col - 1) * channels] + columnSum[col];
								}
							}
						}
					}

					delete[] columnSum;
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
			/// for image(w*h), calculate integral mapping((w+1)*(h+1)), first row and first col are zeros in integral mapping
			/// </summary>
			/// <param name="src">image memory::tensor</param>
			/// <param name="dst">integral mapping memory::tensor</param>
			template <typename Stype, typename Dtype>
			static void fast_integral_cpu(const memory::tensor<Stype> &src, memory::tensor<Dtype>& dst)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_num_offset = channels * height * width;
				int dst_num_offset = channels * (height + 1) * (width + 1);

				if (src.order() == memory::NCHW)
				{
					dst = memory::tensor<Dtype>(std::vector<int>{num, channels, height + 1, width + 1}, src.device(), src.order());
					const Stype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();
					memset(dst_data, (Dtype)0, num * channels * (height + 1) * (width + 1) * sizeof(Dtype));

					// sum of each column
					Dtype *columnSum = new Dtype[width + 1];

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						memset(columnSum, Dtype(0), (width + 1) * sizeof(Dtype));

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_offset = ch * height * width;
							int dst_offset = ch * (height + 1) * (width + 1);

							// calculate integral of the first nonzero_row(second row) 
							unsigned fist_nonzero_row_index = dst_offset + (width + 1);
							for (int col = 1; col < width + 1; ++col)
							{
								columnSum[col] = (Dtype)src_data[src_n_offset + src_offset + col - 1];
								dst_data[dst_n_offset + fist_nonzero_row_index + col] = (Dtype)src_data[src_n_offset + src_offset + col - 1];
								dst_data[dst_n_offset + fist_nonzero_row_index + col] += dst_data[dst_n_offset + fist_nonzero_row_index + col - 1];
							}

							for (int row = 2; row < height + 1; ++row)
							{
								int src_row_offset = (row - 1) * width;
								int dst_row_offset = row * (width + 1);

								for (int col = 1; col < width + 1; ++col)
								{
									columnSum[col] += (Dtype)src_data[src_n_offset + src_offset + src_row_offset + col - 1];
									dst_data[dst_n_offset + dst_offset + dst_row_offset + col] = dst_data[dst_n_offset + dst_offset + dst_row_offset + col - 1] + columnSum[col];
								}
							}
						}
					}

					delete[] columnSum;
				}
				else if (src.order() == memory::NHWC)
				{
					dst = memory::tensor<Dtype>(std::vector<int>{num, height + 1, width + 1, channels}, src.device(), src.order());
					const Stype* src_data = src.cpu_data();
					Dtype* dst_data = dst.mutable_cpu_data();
					memset(dst_data, (Dtype)0, num * channels * (height + 1) * (width + 1) * sizeof(Dtype));

					// sum of each column
					Dtype *columnSum = new Dtype[width + 1];

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						memset(columnSum, Dtype(0), (width + 1) * sizeof(Dtype));

						for (int ch = 0; ch < channels; ++ch)
						{
							// calculate integral of the first nonzero_row(second row) 
							unsigned fist_nonzero_row_index = (width + 1) * channels + ch;
							for (int col = 1; col < width + 1; ++col)
							{
								columnSum[col] = (Dtype)src_data[src_n_offset + (col - 1) * channels + ch];
								dst_data[dst_n_offset + fist_nonzero_row_index + col * channels] = (Dtype)src_data[src_n_offset + (col - 1) * channels + ch];
								dst_data[dst_n_offset + fist_nonzero_row_index + col * channels] += dst_data[dst_n_offset + fist_nonzero_row_index + (col - 1) * channels];
							}

							for (int row = 2; row < height + 1; ++row)
							{
								int src_row_offset = (row - 1) * width * channels + ch;
								int dst_row_offset = row * (width + 1) * channels + ch;

								for (int col = 1; col < width + 1; ++col)
								{
									columnSum[col] += (Dtype)src_data[src_n_offset + src_row_offset + (col - 1) * channels];
									dst_data[dst_n_offset + dst_row_offset + col * channels] = dst_data[dst_n_offset + dst_row_offset + (col - 1) * channels] + columnSum[col];
								}
							}
						}
					}

					delete[] columnSum;
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}



			/// <summary>
            /// calculate histogram, gray image required
            /// </summary>
            /// <param name="src">original memory::tensor</param>
            /// <param name="dst">new memory::tensor</param>
			static void calc_hist_cpu(const std::shared_ptr<memory::tensor<unsigned char>> &src, std::shared_ptr<memory::tensor<float>>& dst, int dimension = 59)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				CHECK_EQ(src->channels(), 1);
				int height = src->height();
				int width = src->width();
				int offset = height * width;

				std::shared_ptr<memory::tensor<int>> dst_int;
				dst_int.reset(new memory::tensor<int>(std::vector<int>{num, 1, 1, dimension}, -1, src->order()));			
				int *dst_int_data = dst_int->mutable_cpu_data();

				const unsigned char* src_data = src->cpu_data();

				for (int n = 0; n < num; n++)
				{
					int *dst_int_data_num = dst_int_data + n * dimension;

					//Count the number of pixels in each grayscale
					for (int i = 0; i < offset; i++)
					{
						int value = static_cast<int>(src_data[n * offset + i]);
						dst_int_data_num[value]++;
					}
				}

				std::vector<int> max_vals(num), min_vals(num);
				for (int n = 0; n < num; n++)
				{
					max_vals[n] = INT_MIN;
					min_vals[n] = INT_MAX;
				}

				for (int n = 0; n < num; n++)
				{
					int *dst_int_data_num = dst_int_data + n * dimension;

					for (int i = 0; i < dimension; i++)
					{
						if (dst_int_data_num[i] > max_vals[n])
						{
							max_vals[n] = dst_int_data_num[i];
						}

						if (dst_int_data_num[i] < min_vals[n])
						{
							min_vals[n] = dst_int_data_num[i];
						}
					}
				}

				dst.reset(new memory::tensor<float>(std::vector<int>{num, 1, 1, dimension}, src->device(), src->order()));
				float *dst_float_data = dst->mutable_cpu_data();

				for (int n = 0; n < num; n++)
				{
					float *dst_float_data_num = dst_float_data + n * dimension;
					int *dst_int_data_num = dst_int_data + n * dimension;

					for (int i = 0; i < dimension; i++)
					{
						dst_float_data_num[i] = float(dst_int_data_num[i] - min_vals[n]) / (max_vals[n] - min_vals[n]);
					}
				}

				return;
			}



			/// <summary>
			/// equalize histogram, gray image required
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename Dtype>
			static void equalize_hist_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				CHECK_EQ(src->channels(), 1);
				int height = src->height();
				int width = src->width();
				int offset = height * width;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				for (int n = 0; n < num; n++)
				{
					int gray_value[256] = { 0 };
					float probability_distribution[256] = { 0 };
					float accumulate_probability_distribution[256] = { 0 };
					int normalized_gray_value[256] = { 0 };

					//Count the number of pixels in each grayscale
					for (int i = 0; i < offset; i++)
					{
						int value = static_cast<unsigned char>(src_data[n * offset + i]);
						gray_value[value]++;
					}

					for (int i = 0; i < 256; i++)
					{
						probability_distribution[i] = static_cast<float>(gray_value[i]) / offset;

						if (i > 0)
						{
							accumulate_probability_distribution[i] = accumulate_probability_distribution[i - 1] + probability_distribution[i];
						}
						else
						{
							accumulate_probability_distribution[0] = probability_distribution[0];
						}

						normalized_gray_value[i] = static_cast<unsigned char>(255 * accumulate_probability_distribution[i] + 0.5);
					}

					for (int i = 0; i < offset; i++)
					{
						dst_data[n * offset + i] = Dtype(normalized_gray_value[static_cast<unsigned char>(src_data[n * offset + i])]);
					}
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// equalize histogram, gray image required
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename Dtype>
			static void equalize_hist_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype>& dst)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				CHECK_EQ(src.channels(), 1);
				int height = src.height();
				int width = src.width();
				int offset = height * width;

				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());				
				Dtype* dst_data = dst_temp.mutable_cpu_data();
				const Dtype* src_data = src.cpu_data();

				for (int n = 0; n < num; n++)
				{
					int gray_value[256] = { 0 };
					float probability_distribution[256] = { 0 };
					float accumulate_probability_distribution[256] = { 0 };
					int normalized_gray_value[256] = { 0 };

					//Count the number of pixels in each grayscale
					for (int i = 0; i < offset; i++)
					{
						int value = static_cast<unsigned char>(src_data[n * offset + i]);
						gray_value[value]++;
					}

					for (int i = 0; i < 256; i++)
					{
						probability_distribution[i] = static_cast<float>(gray_value[i]) / offset;

						if (i > 0)
						{
							accumulate_probability_distribution[i] = accumulate_probability_distribution[i - 1] + probability_distribution[i];
						}
						else
						{
							accumulate_probability_distribution[0] = probability_distribution[0];
						}

						normalized_gray_value[i] = static_cast<unsigned char>(255 * accumulate_probability_distribution[i] + 0.5);
					}

					for (int i = 0; i < offset; i++)
					{
						dst_data[n * offset + i] = Dtype(normalized_gray_value[static_cast<unsigned char>(src_data[n * offset + i])]);
					}
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// split multi-channel image to single-channel image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst_vector">array of single-channel image memory::tensor</param>
			template <typename Dtype>
			static void split_channel_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::vector<std::shared_ptr<memory::tensor<Dtype>>>& dst_vector)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int src_offset = height * width;
				int num_offset = channels * height * width;

				const Dtype* src_data = src->cpu_data();
				dst_vector.clear();

				if (src->order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * src_offset;
							std::shared_ptr<memory::tensor<Dtype>> temp_ptr;
							temp_ptr.reset(new memory::tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->order()));
							Dtype* temp_data = temp_ptr->mutable_cpu_data();
							std::memcpy((void*)temp_data, (void*)(src_data + n_offset + channel_offset), src_offset * sizeof(Dtype));
							dst_vector.push_back(temp_ptr);
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					dst_vector.resize(num * channels);
					std::vector<Dtype *> ptr_arr(num * channels);

					for (int n = 0; n < num; n++)
					{
						for (int ch = 0; ch < channels; ++ch)
						{
							dst_vector.at(n * channels + ch).reset(new memory::tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->order()));
							ptr_arr[n * channels + ch] = dst_vector.at(n * channels + ch)->mutable_cpu_data();
						}
					}

					for (int n = 0; n < num; n++)
					{
						for (int row = 0; row < height; ++row)
						{
							int row_offset = row * width;
							for (int col = 0; col < width; ++col)
							{
								int pos1 = row_offset + col;
								int pos2 = pos1 * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									ptr_arr[n * channels + ch][pos1] = src_data[n * num_offset + pos2 + ch];
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
			/// split multi-channel image to single-channel image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst_vector">array of single-channel image memory::tensor</param>
			template <typename Dtype>
			static void split_channel_cpu(const memory::tensor<Dtype> &src, std::vector<memory::tensor<Dtype>>& dst_vector)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_offset = height * width;
				int num_offset = channels * height * width;

				const Dtype* src_data = src.cpu_data();
				dst_vector.clear();

				if (src.order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * src_offset;
							memory::tensor<Dtype> temp_ptr = memory::tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device(), src.order());
							Dtype* temp_data = temp_ptr.mutable_cpu_data();
							std::memcpy((void*)temp_data, (void*)(src_data + n * num_offset + channel_offset), src_offset * sizeof(Dtype));
							dst_vector.push_back(temp_ptr);
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					dst_vector.resize(num * channels);
					std::vector<Dtype *> ptr_arr(num * channels);

					for (int n = 0; n < num; n++)
					{
						for (int ch = 0; ch < channels; ++ch)
						{
							dst_vector.at(n * channels + ch) = memory::tensor<Dtype>(std::vector<int>{1, height, width, 1}, src.device(), src.order());
							ptr_arr[n * channels + ch] = dst_vector.at(n * channels + ch).mutable_cpu_data();
						}
					}

					for (int n = 0; n < num; n++)
					{
						for (int row = 0; row < height; ++row)
						{
							int row_offset = row * width;
							for (int col = 0; col < width; ++col)
							{
								int pos1 = row_offset + col;
								int pos2 = pos1 * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									ptr_arr[n * channels + ch][pos1] = src_data[n * num_offset + pos2 + ch];
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
			/// merge 3 single-channel images together, to create one 3-channel image 
			/// </summary>
			/// <param name="src_vector">array of tensors, each memory::tensor should share the same height/width/device/order</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename Dtype>
			static void merge_channel_cpu(const std::vector<std::shared_ptr<memory::tensor<Dtype>>> &src_vector, std::shared_ptr<memory::tensor<Dtype>> &dst)
			{
				CHECK_EQ(src_vector.size(), 3);
				int height, width, device;
				memory::orderType order;
				for (int i = 0; i < src_vector.size(); ++i)
				{
					CHECK_EQ(src_vector.at(i)->channels(), 1);
					if (i == 0)
					{
						height = src_vector.at(i)->height();
						width = src_vector.at(i)->width();
						device = src_vector.at(i)->device();
						order = src_vector.at(i)->order();

						if (device >= 0)
						{
							LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
							return;
						}
					}
					else
					{
						if (height != src_vector.at(i)->height() ||
							width != src_vector.at(i)->width() ||
							device != src_vector.at(i)->device() ||
							order != src_vector.at(i)->order())
						{
							LOG(WARNING) << "the element of vector<mat> should have the exact same height/width/device/type.";
							return;
						}
					}
				}

				if (order == memory::NCHW)
				{
					dst.reset(new memory::tensor<Dtype>(std::vector<int>{1, 3, height, width}, device, order));
				}
				else if (order == memory::NHWC)
				{
					dst.reset(new memory::tensor<Dtype>(std::vector<int>{1, height, width, 3}, device, order));
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
			/// merge 3 single-channel images together, to create one 3-channel image
			/// </summary>
			/// <param name="src_vector">array of tensors, each memory::tensor should share the same height/width/device/order</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename Dtype>
			static void merge_channel_cpu(const std::vector<memory::tensor<Dtype>> &src_vector, memory::tensor<Dtype> &dst)
			{
				CHECK_EQ(src_vector.size(), 3);
				int height, width, device;
				memory::orderType order;
				for (int i = 0; i < src_vector.size(); ++i)
				{
					CHECK_EQ(src_vector.at(i)->channels(), 1);
					if (i == 0)
					{
						height = src_vector.at(i).height();
						width = src_vector.at(i).width();
						device = src_vector.at(i).device();
						order = src_vector.at(i).order();

						if (device >= 0)
						{
							LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
							return;
						}
					}
					else
					{
						if (height != src_vector.at(i).height() ||
							width != src_vector.at(i).width() ||
							device != src_vector.at(i).device() ||
							order != src_vector.at(i).order())
						{
							LOG(WARNING) << "the element of vector<mat> should have the exact same height/width/device/type.";
							return;
						}
					}
				}

				if (order == memory::NCHW)
				{
					dst = memory::tensor<Dtype>(std::vector<int>{1, 3, height, width}, device, order);
				}
				else if (order == memory::NHWC)
				{
					dst = memory::tensor<Dtype>(std::vector<int>{1, height, width, 3}, device, order);
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
			/// threshold image data, gray image required
			/// </summary>
			/// <param name="src">original memory::tensor, gray image</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="thresh">threshold value, 128 by default</param>
			/// <param name="maxval">max value, 255 by default</param>
			/// <param name="type">thresholdType: binary(default, set pixel value as maxval when pixel value is greater than thresh, otherwise set pixel value as 0)
			///                               binary_inv(set pixel value as 0 when pixel value is greater than thresh, otherwise set pixel value as maxval)
			///                              small_trunc(set pixel value as thresh when pixel value is smaller than thresh, otherwise remain unchanged)
			///                                big_trunc(set pixel value as thresh when pixel value is greater than thresh, otherwise remain unchanged)
			///                            small_to_zero(set pixel value as 0 when pixel value is smaller than thresh, otherwise remain unchanged)
			///                              big_to_zero(set pixel value as 0 when pixel value is greater than thresh, otherwise remain unchanged)</param>
			template <typename Dtype>
			static void threshold_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				CHECK_EQ(src->channels(), 1);
				int num = src->num();
				int height = src->height();
				int width = src->width();
				int offset = height * width;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_cpu_data();
				const Dtype *src_data = src->cpu_data();

				switch (type)
				{
				case binary:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = src_data[n * offset + j] > (Dtype)thresh ? (Dtype)maxval : 0;
						}
					}

					break;

				case binary_inv:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = src_data[n * offset + j] <= (Dtype)thresh ? (Dtype)maxval : 0;
						}
					}

					break;

				case small_trunc:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = std::max(src_data[n * offset + j], (Dtype)thresh);
						}
					}
					
					break;

				case big_trunc:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = std::min(src_data[n * offset + j], (Dtype)thresh);
						}
					}

					break;

				case small_to_zero:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = src_data[n * offset + j] > (Dtype)thresh ? src_data[n * offset + j] : 0;
						}
					}

					break;

				case big_to_zero:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = src_data[n * offset + j] <= (Dtype)thresh ? src_data[n * offset + j] : 0;
						}
					}

					break;

				default:
					LOG(ERROR) << "Un-support threshold type.";
					break;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// threshold image data, gray image required
			/// </summary>
			/// <param name="src">original memory::tensor, gray image</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="thresh">threshold value, 128 by default</param>
			/// <param name="maxval">max value, 255 by default</param>
			/// <param name="type">thresholdType: binary(default, set pixel value as maxval when pixel value is greater than thresh, otherwise set pixel value as 0)
			///                               binary_inv(set pixel value as 0 when pixel value is greater than thresh, otherwise set pixel value as maxval)
			///                              small_trunc(set pixel value as thresh when pixel value is smaller than thresh, otherwise remain unchanged)
			///                                big_trunc(set pixel value as thresh when pixel value is greater than thresh, otherwise remain unchanged)
			///                            small_to_zero(set pixel value as 0 when pixel value is smaller than thresh, otherwise remain unchanged)
			///                              big_to_zero(set pixel value as 0 when pixel value is greater than thresh, otherwise remain unchanged)</param>
			template <typename Dtype>
			static void threshold_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				CHECK_EQ(src.channels(), 1);
				int num = src.num();				
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());
				Dtype* dst_data = dst_temp.mutable_cpu_data();
				const Dtype *src_data = src.cpu_data();

				switch (type)
				{
				case binary:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = src_data[n * offset + j] > (Dtype)thresh ? (Dtype)maxval : 0;
						}
					}

					break;

				case binary_inv:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = src_data[n * offset + j] <= (Dtype)thresh ? (Dtype)maxval : 0;
						}
					}

					break;

				case small_trunc:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = std::max(src_data[n * offset + j], (Dtype)thresh);
						}
					}

					break;

				case big_trunc:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = std::min(src_data[n * offset + j], (Dtype)thresh);
						}
					}

					break;

				case small_to_zero:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = src_data[n * offset + j] > (Dtype)thresh ? src_data[n * offset + j] : 0;
						}
					}

					break;

				case big_to_zero:
					for (int n = 0; n < num; n++)
					{
						for (int j = 0; j < offset; ++j)
						{
							dst_data[n * offset + j] = src_data[n * offset + j] <= (Dtype)thresh ? src_data[n * offset + j] : 0;
						}
					}

					break;

				default:
					LOG(ERROR) << "Un-support threshold type.";
					break;
				}

				dst = dst_temp.clone();
			}






			/// <summary>
			/// warp affine image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="src_point">3 points before warp affine</param>
			/// <param name="dst_point">3 points after warp affine, get transformation matrix using src_point and dst_point</param>
			/// <param name="fill_pixel_value">fill blank area with fill_pixel_value, zero by default</param>
			/// <param name="type">interpolationType: Nearest / Bilinear(default)</param>
			template <typename Dtype, typename Ptype>
			static void warp_affine_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst,
				const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, Dtype fill_pixel_value = 0, interpolationType type = Bilinear)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (src_point.size() != 3 || dst_point.size() != 3)
				{
					LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;
				unsigned maxIndex = height * width * channels - 1;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));				
				Dtype* dst_data = dst_temp->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				//AX=B, expand A to 6*6, expand B to 6*1
				std::vector<std::vector<float> > A;
				std::vector<float> B, X;
				A.resize(6);
				for (int i = 0; i < 6; i = i + 2)
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

				for (int i = 0; i < 3; i++)
				{
					B.push_back(float(src_point[i].x));
					B.push_back(float(src_point[i].y));
				}

				X = math_functions::gauss_all(A, B);

				if (src->order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										dst_data[n_offset + dst_index] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[n_offset + dst_index] = src_data[n_offset + src_index];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_index), maxIndex);
											unsigned indexB = std::min(unsigned(src_index + 1), maxIndex);
											unsigned indexC = std::min(unsigned(src_index + width), maxIndex);
											unsigned indexD = std::min(unsigned(src_index + width + 1), maxIndex);
											Dtype A = src_data[n_offset + indexA];
											Dtype B = src_data[n_offset + indexB];
											Dtype C = src_data[n_offset + indexC];
											Dtype D = src_data[n_offset + indexD];

											dst_data[n_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else if (src->order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										dst_data[n_offset + dst_pos2 + ch] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[n_offset + dst_pos2 + ch] = src_data[n_offset + src_pos1 + ch];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos1 + ch), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos1 + channels + ch), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos1 + width * channels + ch), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos1 + (width + 1) * channels + ch), maxIndex);
											Dtype A = src_data[n_offset + indexA];
											Dtype B = src_data[n_offset + indexB];
											Dtype C = src_data[n_offset + indexC];
											Dtype D = src_data[n_offset + indexD];

											dst_data[n_offset + dst_pos2 + ch] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// warp affine image
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="src_point">3 points before warp affine</param>
			/// <param name="dst_point">3 points after warp affine, get transformation matrix using src_point and dst_point</param>
			/// <param name="fill_pixel_value">fill blank area with fill_pixel_value, zero by default</param>
			/// <param name="type">interpolationType: Nearest / Bilinear(default)</param>
			template <typename Dtype, typename Ptype>
			static void warp_affine_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst,
				const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, Dtype fill_pixel_value = 0, interpolationType type = Bilinear)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (src_point.size() != 3 || dst_point.size() != 3)
				{
					LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;
				unsigned maxIndex = height * width * channels - 1;

				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());
				Dtype* dst_data = dst_temp.mutable_cpu_data();
				const Dtype* src_data = src.cpu_data();

				//AX=B, expand A to 6*6, expand B to 6*1
				std::vector<std::vector<float> > A;
				std::vector<float> B, X;
				A.resize(6);
				for (int i = 0; i < 6; i = i + 2)
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

				for (int i = 0; i < 3; i++)
				{
					B.push_back(float(src_point[i].x));
					B.push_back(float(src_point[i].y));
				}

				X = math_functions::gauss_all(A, B);

				if (src.order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										dst_data[n_offset + dst_index] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[n_offset + dst_index] = src_data[n_offset + src_index];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_index), maxIndex);
											unsigned indexB = std::min(unsigned(src_index + 1), maxIndex);
											unsigned indexC = std::min(unsigned(src_index + width), maxIndex);
											unsigned indexD = std::min(unsigned(src_index + width + 1), maxIndex);
											Dtype A = src_data[n_offset + indexA];
											Dtype B = src_data[n_offset + indexB];
											Dtype C = src_data[n_offset + indexC];
											Dtype D = src_data[n_offset + indexD];

											dst_data[n_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else if (src.order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										dst_data[n_offset + dst_pos2 + ch] = fill_pixel_value;
									}
									else
									{
										if (type == Nearest)
										{
											dst_data[n_offset + dst_pos2 + ch] = src_data[n_offset + src_pos1 + ch];
										}
										else if (type == Bilinear)
										{
											unsigned indexA = std::min(unsigned(src_pos1 + ch), maxIndex);
											unsigned indexB = std::min(unsigned(src_pos1 + channels + ch), maxIndex);
											unsigned indexC = std::min(unsigned(src_pos1 + width * channels + ch), maxIndex);
											unsigned indexD = std::min(unsigned(src_pos1 + (width + 1) * channels + ch), maxIndex);
											Dtype A = src_data[n_offset + indexA];
											Dtype B = src_data[n_offset + indexB];
											Dtype C = src_data[n_offset + indexC];
											Dtype D = src_data[n_offset + indexD];

											dst_data[n_offset + dst_pos2 + ch] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// gaussian blur
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="ksize">size of gaussian kernel, odd value required, 3 by default</param>
			template <typename Dtype>
			static void gaussian_blur_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, int ksize = 3)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (ksize % 2 != 1)
				{
					LOG(WARNING) << "convolution kernel: width and height should be odd.";
					return;
				}

				if (ksize == 1)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;

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
				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_cpu_data();

				if (src->order() == memory::NCHW)
				{
					std::shared_ptr<memory::tensor<Dtype>> temp;
					temp.reset(new memory::tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
					Dtype* temp_data = temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						//horizontal
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
											sum2 += convolution_kernel[kernel_col + half] * src_data[n * num_offset + index + (col + kernel_col)];
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

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
									dst_data[n * num_offset + pos] = (Dtype)sum2;
								}
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					std::shared_ptr<memory::tensor<Dtype>> temp;
					temp.reset(new memory::tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
					Dtype* temp_data = temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						//horizontal
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
											sum2 += convolution_kernel[kernel_col + half] * src_data[n * num_offset + index + (col + kernel_col) * channels + ch];
										}
									}
									temp_data[index + col * channels + ch] = (Dtype)sum2;
								}
							}
						}


						//vertical
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
									dst_data[n * num_offset + pos + ch] = (Dtype)sum2;
								}
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// gaussian blur
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="ksize">size of gaussian kernel, odd value required, 3 by default</param>
			template <typename Dtype>
			static void gaussian_blur_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst, int ksize = 3)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (ksize % 2 != 1)
				{
					LOG(WARNING) << "convolution kernel: width and height should be odd.";
					return;
				}

				if (ksize == 1)
				{
					dst = src.clone();
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;

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
				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());
				Dtype* dst_data = dst_temp.mutable_cpu_data();

				if (src.order() == memory::NCHW)
				{
					memory::tensor<Dtype> temp = memory::tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
					Dtype* temp_data = temp.mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						//horizontal
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
											sum2 += convolution_kernel[kernel_col + half] * src_data[n * num_offset + index + (col + kernel_col)];
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

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
									dst_data[n * num_offset + pos] = (Dtype)sum2;
								}
							}
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					memory::tensor<Dtype> temp = memory::tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
					Dtype* temp_data = temp.mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						//horizontal
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
											sum2 += convolution_kernel[kernel_col + half] * src_data[n * num_offset + index + (col + kernel_col) * channels + ch];
										}
									}
									temp_data[index + col * channels + ch] = (Dtype)sum2;
								}
							}
						}


						//vertical
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
									dst_data[n * num_offset + pos + ch] = (Dtype)sum2;
								}
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// mean value blur
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="ksize">size of blur kernel, odd value required, 3 by default</param>
			template <typename Dtype>
			static void mean_value_blur_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, int ksize = 3)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (ksize % 2 != 1)
				{
					LOG(WARNING) << "convolution kernel: width and height should be odd.";
					return;
				}

				if (ksize == 1)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;

				int half = (ksize - 1) * 0.5;
				std::vector<double> convolution_kernel(ksize);
				for (int col = 0; col < ksize; ++col)
				{
					convolution_kernel[col] = double(1) / ksize;
				}

				const Dtype* src_data = src->cpu_data();
				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_cpu_data();

				if (src->order() == memory::NCHW)
				{
					std::shared_ptr<memory::tensor<Dtype>> temp;
					temp.reset(new memory::tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
					Dtype* temp_data = temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						//horizontal
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
											sum2 += convolution_kernel[kernel_col + half] * src_data[n * num_offset + index + (col + kernel_col)];
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

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
									dst_data[n * num_offset + pos] = (Dtype)sum2;
								}
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					std::shared_ptr<memory::tensor<Dtype>> temp;
					temp.reset(new memory::tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
					Dtype* temp_data = temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						//horizontal
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
											sum2 += convolution_kernel[kernel_col + half] * src_data[n * num_offset + index + (col + kernel_col) * channels + ch];
										}
									}
									temp_data[index + col * channels + ch] = (Dtype)sum2;
								}
							}
						}


						//vertical
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
									dst_data[n * num_offset + pos + ch] = (Dtype)sum2;
								}
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// mean value blur
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="ksize">size of blur kernel, odd value required, 3 by default</param>
			template <typename Dtype>
			static void mean_value_blur_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst, int ksize = 3)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (ksize % 2 != 1)
				{
					LOG(WARNING) << "convolution kernel: width and height should be odd.";
					return;
				}

				if (ksize == 1)
				{
					dst = src.clone();
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;

				int half = (ksize - 1) * 0.5;
				std::vector<double> convolution_kernel(ksize);
				for (int col = 0; col < ksize; ++col)
				{
					convolution_kernel[col] = double(1) / ksize;
				}

				const Dtype* src_data = src.cpu_data();
				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());
				Dtype* dst_data = dst_temp.mutable_cpu_data();

				if (src.order() == memory::NCHW)
				{
					std::shared_ptr<memory::tensor<Dtype>> temp;
					temp.reset(new memory::tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order()));
					Dtype* temp_data = temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						//horizontal
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
											sum2 += convolution_kernel[kernel_col + half] * src_data[n * num_offset + index + (col + kernel_col)];
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

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
									dst_data[n * num_offset + pos] = (Dtype)sum2;
								}
							}
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					std::shared_ptr<memory::tensor<Dtype>> temp;
					temp.reset(new memory::tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order()));
					Dtype* temp_data = temp->mutable_cpu_data();

					for (int n = 0; n < num; n++)
					{
						//horizontal
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
											sum2 += convolution_kernel[kernel_col + half] * src_data[n * num_offset + index + (col + kernel_col) * channels + ch];
										}
									}
									temp_data[index + col * channels + ch] = (Dtype)sum2;
								}
							}
						}


						//vertical
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
									dst_data[n * num_offset + pos + ch] = (Dtype)sum2;
								}
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// calculate gradient in horizontal / vertical direction
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="dx"> 1 by default, calculate horizontal gradient when dx is greater than 0, otherwise do nothing</param>
			/// <param name="dy"> 1 by default, calculate vertical gradient when dy is greater than 0, otherwise do nothing</param>
			template <typename Dtype>
			static void sobel_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, int dx = 1, int dy = 1)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (dx == 0 && dy == 0)
				{
					LOG(WARNING) << "dx, dy cannot be zero simultaneously.";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				if (src->order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										sumx = src_data[n_offset + posSub + 1] + 2 * src_data[n_offset + pos + 1] + src_data[n_offset + posAdd + 1]
											- src_data[n_offset + posSub - 1] - 2 * src_data[n_offset + pos - 1] - src_data[n_offset + posAdd - 1];
									}

									if (dy > 0)
									{
										sumy = src_data[n_offset + posSub - 1] + 2 * src_data[n_offset + posSub] + src_data[n_offset + posSub + 1]
											- src_data[n_offset + posAdd - 1] - 2 * src_data[n_offset + posAdd] - src_data[n_offset + posAdd + 1];
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

									dst_data[n_offset + pos] = total;
								}
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										sumx = src_data[n_offset + posSub + channels + ch] + 2 * src_data[n_offset + pos + channels + ch] + src_data[n_offset + posAdd + channels + ch]
											- src_data[n_offset + posSub - channels + ch] - 2 * src_data[n_offset + pos - channels + ch] - src_data[n_offset + posAdd - channels + ch];
									}

									if (dy > 0)
									{
										sumy = src_data[n_offset + posSub - channels + ch] + 2 * src_data[n_offset + posSub + ch] + src_data[n_offset + posSub + channels + ch]
											- src_data[n_offset + posAdd - channels + ch] - 2 * src_data[n_offset + posAdd + ch] - src_data[n_offset + posAdd + channels + ch];
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

									dst_data[n_offset + pos + ch] = total;
								}
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// calculate gradient in horizontal / vertical direction
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="dx"> 1 by default, calculate horizontal gradient when dx is greater than 0, otherwise do nothing</param>
			/// <param name="dy"> 1 by default, calculate vertical gradient when dy is greater than 0, otherwise do nothing</param>
			template <typename Dtype>
			static void sobel_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst, int dx = 1, int dy = 1)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (dx == 0 && dy == 0)
				{
					LOG(WARNING) << "dx, dy cannot be zero simultaneously.";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;

				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());
				Dtype* dst_data = dst_temp.mutable_cpu_data();
				const Dtype* src_data = src.cpu_data();

				if (src.order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										sumx = src_data[n_offset + posSub + 1] + 2 * src_data[n_offset + pos + 1] + src_data[n_offset + posAdd + 1]
											- src_data[n_offset + posSub - 1] - 2 * src_data[n_offset + pos - 1] - src_data[n_offset + posAdd - 1];
									}

									if (dy > 0)
									{
										sumy = src_data[n_offset + posSub - 1] + 2 * src_data[n_offset + posSub] + src_data[n_offset + posSub + 1]
											- src_data[n_offset + posAdd - 1] - 2 * src_data[n_offset + posAdd] - src_data[n_offset + posAdd + 1];
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

									dst_data[n_offset + pos] = total;
								}
							}
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * num_offset;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
										sumx = src_data[n_offset + posSub + channels + ch] + 2 * src_data[n_offset + pos + channels + ch] + src_data[n_offset + posAdd + channels + ch]
											- src_data[n_offset + posSub - channels + ch] - 2 * src_data[n_offset + pos - channels + ch] - src_data[n_offset + posAdd - channels + ch];
									}

									if (dy > 0)
									{
										sumy = src_data[n_offset + posSub - channels + ch] + 2 * src_data[n_offset + posSub + ch] + src_data[n_offset + posSub + channels + ch]
											- src_data[n_offset + posAdd - channels + ch] - 2 * src_data[n_offset + posAdd + ch] - src_data[n_offset + posAdd + channels + ch];
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

									dst_data[n_offset + pos + ch] = total;
								}
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				dst = dst_temp.clone();
			}



			/// <summary>
			/// dilate or erode image data
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="type">morphType: Dilate(default) / Erode</param>
			/// <param name="ksize">size of square kernel, odd value required, 3 by default</param>
			template <typename Dtype>
			static void morph_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, morphType type = Dilate, int ksize = 3)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (ksize % 2 != 1)
				{
					LOG(WARNING) << "ksize should be odd.";
					return;
				}

				if (ksize == 1)
				{
					dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int offset = height * width;
				int num_offset = channels * height * width;
				int half = (ksize - 1) * 0.5;

				std::shared_ptr<memory::tensor<Dtype>> dst_temp;
				dst_temp.reset(new memory::tensor<Dtype>(src->data_shape(), src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				if (src->order() == memory::NCHW)
				{
					if (type == morphType::Dilate)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
												if (src_data[n_offset + pos3] > max)
												{
													max = src_data[n_offset + pos3];
												}
											}
										}
										dst_data[n_offset + pos1] = max;
									}
								}
							}
						}
					}
					else if (type == morphType::Erode)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
												if (src_data[n_offset + pos3] < min)
												{
													min = src_data[n_offset + pos3];
												}
											}
										}
										dst_data[n_offset + pos1] = min;
									}
								}
							}
						}
					}
					else
					{
						LOG(WARNING) << "unsupported type.";
					}
				}
				else if (src->order() == memory::NHWC)
				{
					if (type == morphType::Dilate)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
												if (src_data[n_offset + pos3] > max)
												{
													max = src_data[n_offset + pos3];
												}
											}
										}
										dst_data[n_offset + pos1 + ch] = max;
									}
								}
							}
						}
					}
					else if (type == morphType::Erode)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
												if (src_data[n_offset + pos3] < min)
												{
													min = src_data[n_offset + pos3];
												}
											}
										}
										dst_data[n_offset + pos1 + ch] = min;
									}
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

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			/// <summary>
			/// dilate or erode image data
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			/// <param name="type">morphType: Dilate(default) / Erode</param>
			/// <param name="ksize">size of square kernel, odd value required, 3 by default</param>
			template <typename Dtype>
			static void morph_cpu(const memory::tensor<Dtype> &src, memory::tensor<Dtype> &dst, morphType type = Dilate, int ksize = 3)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				if (ksize % 2 != 1)
				{
					LOG(WARNING) << "ksize should be odd.";
					return;
				}

				if (ksize == 1)
				{
					dst = src.clone();
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int offset = height * width;
				int num_offset = channels * height * width;
				int half = (ksize - 1) * 0.5;

				memory::tensor<Dtype> dst_temp = memory::tensor<Dtype>(src.data_shape(), src.device(), src.order());
				Dtype* dst_data = dst_temp.mutable_cpu_data();
				const Dtype* src_data = src.cpu_data();

				if (src.order() == memory::NCHW)
				{
					if (type == morphType::Dilate)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
												if (src_data[n_offset + pos3] > max)
												{
													max = src_data[n_offset + pos3];
												}
											}
										}
										dst_data[n_offset + pos1] = max;
									}
								}
							}
						}
					}
					else if (type == morphType::Erode)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
							for (int ch = 0; ch < channels; ++ch)
							{
								int channel_offset = ch * offset;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
												if (src_data[n_offset + pos3] < min)
												{
													min = src_data[n_offset + pos3];
												}
											}
										}
										dst_data[n_offset + pos1] = min;
									}
								}
							}
						}
					}
					else
					{
						LOG(WARNING) << "unsupported type.";
					}
				}
				else if (src.order() == memory::NHWC)
				{
					if (type == morphType::Dilate)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
												if (src_data[n_offset + pos3] > max)
												{
													max = src_data[n_offset + pos3];
												}
											}
										}
										dst_data[n_offset + pos1 + ch] = max;
									}
								}
							}
						}
					}
					else if (type == morphType::Erode)
					{
						for (int n = 0; n < num; n++)
						{
							int n_offset = n * num_offset;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
												if (src_data[n_offset + pos3] < min)
												{
													min = src_data[n_offset + pos3];
												}
											}
										}
										dst_data[n_offset + pos1 + ch] = min;
									}
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

				dst = dst_temp.clone();
			}



			template <typename Dtype>
			/// <summary>
			/// for image(w*h), calculate LBP feature((w-2)*(h-2))
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">LBP feature memory::tensor</param>
			/// <param name="map_59">map_59: map 256-dimension LBP-feature to 59-dimension)</param>
			static void lbp_feature_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, bool map_59 = false)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				int src_num_offset = channels * height * width;
				int dst_num_offset = channels * (height - 2) * (width - 2);
				std::shared_ptr<memory::tensor<Dtype>> dst_temp;

				if (src->order() == memory::NCHW)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, height - 2, width - 2}, src->device(), src->order()));
				}
				else if (src->order() == memory::NHWC)
				{
					dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, height - 2, width - 2, channels}, src->device(), src->order()));
				}
				else
				{
					NOT_IMPLEMENTED;
				}
				
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst_temp->mutable_cpu_data();

				if (src->order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						for (int c = 0; c < channels; c++)
						{
							int src_c_offset = c * height * width;
							int dst_c_offset = c * (height - 2) * (width - 2);
							for (int h = 1; h < height - 1; h++)
							{
								int offset = src_c_offset + h * width;
								int offset_plus = offset + width;
								int offset_minus = offset - width;
								for (int w = 1; w < width - 1; w++)
								{
									Dtype center = src_data[src_n_offset + offset + w];
									unsigned char code = 0;
									//code |= (src_data[src_n_offset + offset_minus + w - 1] >= center) << 7;
									//code |= (src_data[src_n_offset + offset_minus + w - 0] >= center) << 6;
									//code |= (src_data[src_n_offset + offset_minus + w + 1] >= center) << 5;
									//code |= (src_data[src_n_offset + offset + w + 1] >= center) << 4;
									//code |= (src_data[src_n_offset + offset_plus + w + 1] >= center) << 3;
									//code |= (src_data[src_n_offset + offset_plus + w + 0] >= center) << 2;
									//code |= (src_data[src_n_offset + offset_plus + w - 1] >= center) << 1;
									//code |= (src_data[src_n_offset + offset + w - 1] >= center) << 0;

									code |= (src_data[src_n_offset + offset_minus + w - 1] > center) << 0;
									code |= (src_data[src_n_offset + offset_minus + w - 0] > center) << 1;
									code |= (src_data[src_n_offset + offset_minus + w + 1] > center) << 2;
									code |= (src_data[src_n_offset + offset + w + 1] > center) << 3;
									code |= (src_data[src_n_offset + offset_plus + w + 1] > center) << 4;
									code |= (src_data[src_n_offset + offset_plus + w + 0] > center) << 5;
									code |= (src_data[src_n_offset + offset_plus + w - 1] > center) << 6;
									code |= (src_data[src_n_offset + offset + w - 1] > center) << 7;

									if (map_59)
									{
										dst_data[dst_n_offset + dst_c_offset + (h - 1) * (width - 2) + w - 1] = static_cast<Dtype>(LBPMAP[0][code]);
									}
									else
									{
										dst_data[dst_n_offset + dst_c_offset + (h - 1) * (width - 2) + w - 1] = static_cast<Dtype>(code);
									}
								}
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						for (int h = 1; h < height - 1; h++)
						{
							int offset = h * width * channels;
							int offset_plus = offset + width * channels;
							int offset_minus = offset - width * channels;
							for (int w = 1; w < width - 1; w++)
							{
								for (int c = 0; c < channels; c++)
								{
									Dtype center = src_data[src_n_offset + offset + w * channels + c];
									unsigned char code = 0;

									//code |= (src_data[src_n_offset + offset_minus + (w - 1) * channels + c] >= center) << 7;
									//code |= (src_data[src_n_offset + offset_minus + (w - 0) * channels + c] >= center) << 6;
									//code |= (src_data[src_n_offset + offset_minus + (w + 1) * channels + c] >= center) << 5;
									//code |= (src_data[src_n_offset + offset + (w + 1) * channels + c] >= center) << 4;
									//code |= (src_data[src_n_offset + offset_plus + (w + 1) * channels + c] >= center) << 3;
									//code |= (src_data[src_n_offset + offset_plus + (w + 0) * channels + c] >= center) << 2;
									//code |= (src_data[src_n_offset + offset_plus + (w - 1) * channels + c] >= center) << 1;
									//code |= (src_data[src_n_offset + offset + (w - 1) * channels + c] >= center) << 0;

									code |= (src_data[src_n_offset + offset_minus + (w - 1) * channels + c] > center) << 0;
									code |= (src_data[src_n_offset + offset_minus + (w - 0) * channels + c] > center) << 1;
									code |= (src_data[src_n_offset + offset_minus + (w + 1) * channels + c] > center) << 2;
									code |= (src_data[src_n_offset + offset + (w + 1) * channels + c] > center) << 3;
									code |= (src_data[src_n_offset + offset_plus + (w + 1) * channels + c] > center) << 4;
									code |= (src_data[src_n_offset + offset_plus + (w + 0) * channels + c] > center) << 5;
									code |= (src_data[src_n_offset + offset_plus + (w - 1) * channels + c] > center) << 6;
									code |= (src_data[src_n_offset + offset + (w - 1) * channels + c] > center) << 7;

									if (map_59)
									{
										dst_data[dst_n_offset + ((h - 1) * (width - 2) + (w - 1)) * channels + c] = static_cast<Dtype>(LBPMAP[0][code]);
									}
									else
									{
										dst_data[dst_n_offset + ((h - 1) * (width - 2) + (w - 1)) * channels + c] = static_cast<Dtype>(code);
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

				dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
			}



			template <typename Dtype>
			/// <summary>
			/// for image(w*h), calculate LBP feature((w-2)*(h-2))
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">LBP feature memory::tensor</param>
			/// <param name="map_59">map_59: map 256-dimension LBP-feature to 59-dimension)</param>
			static void lbp_feature_cpu(const memory::tensor<Dtype>& src, memory::tensor<Dtype>& dst, bool map_59 = false)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				int src_num_offset = channels * height * width;
				int dst_num_offset = channels * (height - 2) * (width - 2);

				memory::tensor<Dtype> dst_temp;
				if (src.order() == memory::NCHW)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, channels, height - 2, width - 2}, src->device(), src->order());
				}
				else if (src.order() == memory::NHWC)
				{
					dst_temp = memory::tensor<Dtype>(std::vector<int>{num, height - 2, width - 2, channels}, src->device(), src->order());
				}
				else
				{
					NOT_IMPLEMENTED;
				}
				const Dtype* src_data = src.cpu_data();
				Dtype* dst_data = dst_temp.mutable_cpu_data();

				if (src->order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						for (int c = 0; c < channels; c++)
						{
							int src_c_offset = c * height * width;
							int dst_c_offset = c * (height - 2) * (width - 2);
							for (int h = 1; h < height - 1; h++)
							{
								int offset = src_c_offset + h * width;
								int offset_plus = offset + width;
								int offset_minus = offset - width;
								for (int w = 1; w < width - 1; w++)
								{
									Dtype center = src_data[src_n_offset + offset + w];
									unsigned char code = 0;
									//code |= (src_data[src_n_offset + offset_minus + w - 1] >= center) << 7;
									//code |= (src_data[src_n_offset + offset_minus + w - 0] >= center) << 6;
									//code |= (src_data[src_n_offset + offset_minus + w + 1] >= center) << 5;
									//code |= (src_data[src_n_offset + offset + w + 1] >= center) << 4;
									//code |= (src_data[src_n_offset + offset_plus + w + 1] >= center) << 3;
									//code |= (src_data[src_n_offset + offset_plus + w + 0] >= center) << 2;
									//code |= (src_data[src_n_offset + offset_plus + w - 1] >= center) << 1;
									//code |= (src_data[src_n_offset + offset + w - 1] >= center) << 0;

									code |= (src_data[src_n_offset + offset_minus + w - 1] > center) << 0;
									code |= (src_data[src_n_offset + offset_minus + w - 0] > center) << 1;
									code |= (src_data[src_n_offset + offset_minus + w + 1] > center) << 2;
									code |= (src_data[src_n_offset + offset + w + 1] > center) << 3;
									code |= (src_data[src_n_offset + offset_plus + w + 1] > center) << 4;
									code |= (src_data[src_n_offset + offset_plus + w + 0] > center) << 5;
									code |= (src_data[src_n_offset + offset_plus + w - 1] > center) << 6;
									code |= (src_data[src_n_offset + offset + w - 1] > center) << 7;

									if (map_59)
									{
										dst_data[dst_n_offset + dst_c_offset + (h - 1) * (width - 2) + w - 1] = static_cast<Dtype>(LBPMAP[0][code]);
									}
									else
									{
										dst_data[dst_n_offset + dst_c_offset + (h - 1) * (width - 2) + w - 1] = static_cast<Dtype>(code);
									}									
								}
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						for (int h = 1; h < height - 1; h++)
						{
							int offset = h * width * channels;
							int offset_plus = offset + width * channels;
							int offset_minus = offset - width * channels;
							for (int w = 1; w < width - 1; w++)
							{
								for (int c = 0; c < channels; c++)
								{
									Dtype center = src_data[src_n_offset + offset + w * channels + c];
									unsigned char code = 0;
									//code |= (src_data[src_n_offset + offset_minus + (w - 1) * channels + c] >= center) << 7;
									//code |= (src_data[src_n_offset + offset_minus + (w - 0) * channels + c] >= center) << 6;
									//code |= (src_data[src_n_offset + offset_minus + (w + 1) * channels + c] >= center) << 5;
									//code |= (src_data[src_n_offset + offset + (w + 1) * channels + c] >= center) << 4;
									//code |= (src_data[src_n_offset + offset_plus + (w + 1) * channels + c] >= center) << 3;
									//code |= (src_data[src_n_offset + offset_plus + (w + 0) * channels + c] >= center) << 2;
									//code |= (src_data[src_n_offset + offset_plus + (w - 1) * channels + c] >= center) << 1;
									//code |= (src_data[src_n_offset + offset + (w - 1) * channels + c] >= center) << 0;

									code |= (src_data[src_n_offset + offset_minus + (w - 1) * channels + c] > center) << 0;
									code |= (src_data[src_n_offset + offset_minus + (w - 0) * channels + c] > center) << 1;
									code |= (src_data[src_n_offset + offset_minus + (w + 1) * channels + c] > center) << 2;
									code |= (src_data[src_n_offset + offset + (w + 1) * channels + c] > center) << 3;
									code |= (src_data[src_n_offset + offset_plus + (w + 1) * channels + c] > center) << 4;
									code |= (src_data[src_n_offset + offset_plus + (w + 0) * channels + c] > center) << 5;
									code |= (src_data[src_n_offset + offset_plus + (w - 1) * channels + c] > center) << 6;
									code |= (src_data[src_n_offset + offset + (w - 1) * channels + c] > center) << 7;

									if (map_59)
									{
										dst_data[dst_n_offset + ((h - 1) * (width - 2) + (w - 1)) * channels + c] = static_cast<Dtype>(LBPMAP[0][code]);
									}
									else
									{
										dst_data[dst_n_offset + ((h - 1) * (width - 2) + (w - 1)) * channels + c] = static_cast<Dtype>(code);
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

				dst = dst_temp.clone();
			}



			template <typename Dtype>
			/// <summary>
			/// calculate LBP feature with block
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">LBP feature memory::tensor</param>
			/// <param name="block_h">block height</param>
			/// <param name="block_w">block width</param>
			/// <param name="stride_h">stride height</param>
			/// <param name="stride_w">stride width</param>
			static void mblbp_feature_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, int block_h,
				int block_w, int stride_h, int stride_w)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				CHECK_GE(block_h, 1);
				CHECK_GE(block_w, 1);
				int num = src->num();
				int channels = src->channels();
				int height = src->height();
				int width = src->width();
				CHECK_GE(height, 3 * block_h);
				CHECK_GE(width, 3 * block_w);
				const Dtype* src_data = src->cpu_data();
				Dtype* dst_data = dst->mutable_cpu_data();

				std::shared_ptr<memory::tensor<Dtype>> src_tensor = std::make_shared<memory::tensor<Dtype>>(src);
				std::shared_ptr<memory::tensor<Dtype>> integral_tensor = std::make_shared<memory::tensor<Dtype>>(memory::tensor<Dtype>::memory::tensor(src->order()));
				fast_integral_cpu(src_tensor, integral_tensor);
				Dtype* integral_data = integral_tensor->mutable_cpu_data();

				int dst_height = dst->height();
				int dst_width = dst->width();
				int dst_num_offset = channels * dst_height * dst_width;
				int integral_num_offset = channels * (height + 1) * (width + 1);

				if (src->order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int dst_n_offset = n * dst_num_offset;
						int integral_n_offset = n * integral_num_offset;
						for (int c = 0; c < channels; c++)
						{
							int src_offset = c * height * width;
							int dst_offset = c * dst_height * dst_width;
							int integral_offset = c * (height + 1) * (width + 1);
							for (int h = 0; h < height - 3 * block_h + 1; h += stride_h)
							{
								int dst_sub_offset = h / stride_h * dst_width;
								for (int w = 0; w < width - 3 * block_w + 1; w += stride_w)
								{
									Dtype block_values[9];
									for (int i = 0; i < 9; i++)
									{
										int x1 = w + (i % 3) * block_w;
										int y1 = h + (i / 3) * block_h;
										int x2 = x1 + block_w;
										int y2 = y1 + block_h;
										float A = (float)integral_data[integral_n_offset + integral_offset + y1 * (height + 1) + x1];
										float B = (float)integral_data[integral_n_offset + integral_offset + y1 * (height + 1) + x2];
										float C = (float)integral_data[integral_n_offset + integral_offset + y2 * (height + 1) + x1];
										float D = (float)integral_data[integral_n_offset + integral_offset + y2 * (height + 1) + x2];
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
									dst_data[dst_n_offset + dst_offset + dst_sub_offset + w / stride_w] = Dtype(code);
								}
							}
						}
					}
				}
				else if (src->order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int dst_n_offset = n * dst_num_offset;
						int integral_n_offset = n * integral_num_offset;
						for (int h = 0; h < height - 3 * block_h + 1; h += stride_h)
						{
							int dst_offset = h / stride_h * dst_width * channels;
							for (int w = 0; w < width - 3 * block_w + 1; w += stride_w)
							{
								for (int c = 0; c < channels; c++)
								{
									Dtype block_values[9];
									for (int i = 0; i < 9; i++)
									{
										int x1 = w + (i % 3) * block_w;
										int y1 = h + (i / 3) * block_h;
										int x2 = x1 + block_w;
										int y2 = y1 + block_h;
										float A = (float)integral_data[integral_n_offset + (y1 * (height + 1) + x1) * channels + c];
										float B = (float)integral_data[integral_n_offset + (y1 * (height + 1) + x2) * channels + c];
										float C = (float)integral_data[integral_n_offset + (y2 * (height + 1) + x1) * channels + c];
										float D = (float)integral_data[integral_n_offset + (y2 * (height + 1) + x2) * channels + c];
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
									dst_data[dst_n_offset + dst_offset + w / stride_w * channels + c] = Dtype(code);
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



			template <typename Dtype>
			/// <summary>
			/// calculate LBP feature with block
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">LBP feature memory::tensor</param>
			/// <param name="block_h">block height</param>
			/// <param name="block_w">block width</param>
			/// <param name="stride_h">stride height</param>
			/// <param name="stride_w">stride width</param>
			static void mblbp_feature_cpu(const memory::tensor<Dtype>& src, memory::tensor<Dtype>& dst, int block_h,
				int block_w, int stride_h, int stride_w)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

				int num = src.num();
				CHECK_GE(block_h, 1);
				CHECK_GE(block_w, 1);
				int channels = src.channels();
				int height = src.height();
				int width = src.width();
				CHECK_GE(height, 3 * block_h);
				CHECK_GE(width, 3 * block_w);
				const Dtype* src_data = src.cpu_data();
				Dtype* dst_data = dst.mutable_cpu_data();

				memory::tensor<Dtype> integral_tensor = memory::tensor<Dtype>::memory::tensor(src->order());
				fast_integral_cpu(src, integral_tensor);
				Dtype* integral_data = integral_tensor.mutable_cpu_data();

				int dst_height = dst.height();
				int dst_width = dst.width();
				int dst_num_offset = channels * dst_height * dst_width;
				int integral_num_offset = channels * (height + 1) * (width + 1);

				if (src.order() == memory::NCHW)
				{
					for (int n = 0; n < num; n++)
					{
						int dst_n_offset = n * dst_num_offset;
						int integral_n_offset = n * integral_num_offset;
						for (int c = 0; c < channels; c++)
						{
							int src_offset = c * height * width;
							int dst_offset = c * dst_height * dst_width;
							int integral_offset = c * (height + 1) * (width + 1);
							for (int h = 0; h < height - 3 * block_h + 1; h += stride_h)
							{
								int dst_sub_offset = h / stride_h * dst_width;
								for (int w = 0; w < width - 3 * block_w + 1; w += stride_w)
								{
									Dtype block_values[9];
									for (int i = 0; i < 9; i++)
									{
										int x1 = w + (i % 3) * block_w;
										int y1 = h + (i / 3) * block_h;
										int x2 = x1 + block_w;
										int y2 = y1 + block_h;
										float A = (float)integral_data[integral_n_offset + integral_offset + y1 * (height + 1) + x1];
										float B = (float)integral_data[integral_n_offset + integral_offset + y1 * (height + 1) + x2];
										float C = (float)integral_data[integral_n_offset + integral_offset + y2 * (height + 1) + x1];
										float D = (float)integral_data[integral_n_offset + integral_offset + y2 * (height + 1) + x2];
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
									dst_data[dst_n_offset + dst_offset + dst_sub_offset + w / stride_w] = Dtype(code);
								}
							}
						}
					}
				}
				else if (src.order() == memory::NHWC)
				{
					for (int n = 0; n < num; n++)
					{
						int dst_n_offset = n * dst_num_offset;
						int integral_n_offset = n * integral_num_offset;
						for (int h = 0; h < height - 3 * block_h + 1; h += stride_h)
						{
							int dst_offset = h / stride_h * dst_width * channels;
							for (int w = 0; w < width - 3 * block_w + 1; w += stride_w)
							{
								for (int c = 0; c < channels; c++)
								{
									Dtype block_values[9];
									for (int i = 0; i < 9; i++)
									{
										int x1 = w + (i % 3) * block_w;
										int y1 = h + (i / 3) * block_h;
										int x2 = x1 + block_w;
										int y2 = y1 + block_h;
										float A = (float)integral_data[integral_n_offset + (y1 * (height + 1) + x1) * channels + c];
										float B = (float)integral_data[integral_n_offset + (y1 * (height + 1) + x2) * channels + c];
										float C = (float)integral_data[integral_n_offset + (y2 * (height + 1) + x1) * channels + c];
										float D = (float)integral_data[integral_n_offset + (y2 * (height + 1) + x2) * channels + c];
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
									dst_data[dst_n_offset + dst_offset + w / stride_w * channels + c] = Dtype(code);
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
			/// preprocess memory::tensor
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename DtypeSRC, typename DtypeDST>
			static void preprocess_tensors_cpu(const std::shared_ptr<memory::tensor<DtypeSRC>> &src, std::shared_ptr<memory::tensor<DtypeDST>> &dst, float means[3], float var);



			/// <summary>
			/// preprocess memory::tensor
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">new memory::tensor</param>
			template <typename DtypeSRC, typename DtypeDST>
			static void preprocess_tensors_cpu(const memory::tensor<DtypeSRC> &src, memory::tensor<DtypeDST> &dst, float means[3], float var);


			
			/// <summary>
			/// get ROI(region of interest) from image. Similar to roi_cpu, but more safe, if rectangle exceeds border, fill 0 instead. 
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">ROI memory::tensor</param>
			/// <param name="rect">region of interest</param>
			template <typename Dtype, typename Rtype>
			static void safty_cut_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, rectangle<Rtype>* rect)
			{
				if (src->device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

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
					std::shared_ptr<memory::tensor<Dtype>> temp;
					if (src->order() == memory::NCHW)
					{
						temp.reset(new memory::tensor<Dtype>(
							std::vector<int>{src->num(), src->channels(), src->height() + top + bottom, src->width() + left + right},
							src->device(), src->order()));
					}
					else if (src->order() == memory::NHWC)
					{
						temp.reset(new memory::tensor<Dtype>(
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
			/// get ROI(region of interest) from image. Similar to roi_cpu, but more safe, if rectangle exceeds border, fill 0 instead. 
			/// </summary>
			/// <param name="src">original memory::tensor</param>
			/// <param name="dst">ROI memory::tensor</param>
			/// <param name="rect">region of interest</param>
			template <typename Dtype, typename Rtype>
			static void safty_cut_cpu(const memory::tensor<Dtype>& src, memory::tensor<Dtype>& dst, rectangle<Rtype>* rect)
			{
				if (src.device() >= 0)
				{
					LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
					return;
				}

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
					memory::tensor<Dtype>* temp;
					if (src.order() == memory::NCHW)
					{
						temp = new memory::tensor<Dtype>(
							std::vector<int>{src.num(), src.channels(), src.height() + top + bottom, src.width() + left + right},
							src.device(), src.order());
					}
					else if (src.order() == memory::NHWC)
					{
						temp = new memory::tensor<Dtype>(
							std::vector<int>{src.num(), src.height() + top + bottom, src.width() + left + right, src.channels()},
							src.device(), src.order());
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



			/// <summary>
            /// calculate absolute pixel-subtraction of two images
            /// </summary>
            /// <param name="channel1">first image</param>
            /// <param name="channel2">second image</param>
            /// <param name="result">absolute pixel-subtraction image</param>
			static void absdiff_cpu(const std::shared_ptr<memory::tensor<unsigned char>> &channel1, const std::shared_ptr<memory::tensor<unsigned char>> &channel2, std::shared_ptr<memory::tensor<unsigned char>> &result)
			{
				CHECK_EQ(channel1->num(), channel2->num());
				CHECK_EQ(channel1->channels(), channel2->channels());
				CHECK_EQ(channel1->height(), channel2->height());
				CHECK_EQ(channel1->width(), channel2->width());

				std::shared_ptr<memory::tensor<unsigned char>> sub_value;
				sub_value.reset(new memory::tensor<unsigned char>(channel1->data_shape(), channel1->device(), channel1->order()));

				const unsigned char *channel1_data = channel1->cpu_data();
				const unsigned char *channel2_data = channel2->cpu_data();
				unsigned char *sub_value_data = sub_value->mutable_cpu_data();

				for (size_t i = 0; i < channel1->count(); i++)
				{
					int num1 = channel1_data[i];
					int num2 = channel2_data[i];
					int temp = abs(num1 - num2);

					if (temp > 255)
					{
						temp = 255;
					}
					else if (temp < 0)
					{
						temp = 0;
					}

					sub_value_data[i] = temp;
				}

				result = std::make_shared<memory::tensor<unsigned char>>(sub_value->clone());
			}



			/// <summary>
            /// calculate pixel-sum of two images
            /// </summary>
            /// <param name="channel1">first image</param>
            /// <param name="channel2">second image</param>
            /// <param name="result">pixel-sum image</param>
			static void add_channel_cpu(const std::shared_ptr<memory::tensor<unsigned char>> &channel1, const std::shared_ptr<memory::tensor<unsigned char>> &channel2, std::shared_ptr<memory::tensor<unsigned char>> &result)
			{
				CHECK_EQ(channel1->num(), channel2->num());
				CHECK_EQ(channel1->channels(), channel2->channels());
				CHECK_EQ(channel1->height(), channel2->height());
				CHECK_EQ(channel1->width(), channel2->width());

				std::shared_ptr<memory::tensor<unsigned char>> add_value;
				add_value.reset(new memory::tensor<unsigned char>(channel1->data_shape(), channel1->device(), channel1->order()));

				const unsigned char *channel1_data = channel1->cpu_data();
				const unsigned char *channel2_data = channel2->cpu_data();
				unsigned char *add_value_data = add_value->mutable_cpu_data();

				for (size_t i = 0; i < channel1->count(); i++)
				{
					int num1 = channel1_data[i];
					int num2 = channel2_data[i];
					int temp = abs(num1 + num2);

					if (temp > 255)
					{
						temp = 255;
					}
					else if (temp < 0)
					{
						temp = 0;
					}

					add_value_data[i] = temp;
				}

				result = std::make_shared<memory::tensor<unsigned char>>(add_value->clone());
			}



			/// <summary>
            /// calculate mean-value of input image
            /// </summary>
            /// <param name="array_tensor">image memory::tensor</param>
			static float mean_array(const std::shared_ptr<memory::tensor<unsigned char>> &array_tensor)
			{
				const unsigned char *array_data = array_tensor->cpu_data();
				float sum = 0;
				for (size_t i = 0; i < array_tensor->count(); i++)
				{
					sum += array_data[i];
				}

				return sum / array_tensor->count();
			}

		private:

#ifdef USE_OPENCV

			/// <summary>
			/// convert memory::tensor to cv::Mat, then show image
			/// </summary>
			/// <param name="dst">image memory::tensor</param>
			template <typename Dtype>
			static void showimage(std::shared_ptr<memory::tensor<Dtype>>& dst)
			{
				std::vector<cv::Mat> mat;
				tensor2mat_cpu(dst, mat);
				cv::Mat showmat;
				for (int i = 0; i < mat.size(); i++)
				{
					mat[i].convertTo(showmat, CV_8U);
					cv::imshow("test", showmat);
					cv::waitKey();
				}
			}



			/// <summary>
			/// convert memory::tensor to cv::Mat, then show image
			/// </summary>
			/// <param name="dst">image memory::tensor</param>
			template <typename Dtype>
			static void showimage(memory::tensor<Dtype>& dst)
			{
				std::vector<cv::Mat> mat;
				tensor2mat_cpu(dst, mat);
				cv::Mat showmat;
				for (int i = 0; i < mat.size(); i++)
				{
					mat[i].convertTo(showmat, CV_8U);
					cv::imshow("test", showmat);
					cv::waitKey();
				}
			}



			/// <summary>
			/// get depth according to data type
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
