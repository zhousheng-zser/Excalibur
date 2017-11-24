#include <msclr\marshal_cppstd.h>
#include "ipbbox_net.hpp"
#include "ipts_net.hpp"

using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;

#define MARSHAL_ARRAY(n_array, m_array, n_array_size) \
  auto m_array = gcnew array<float>(n_array_size); \
  pin_ptr<float> pma = &m_array[0]; \
  memcpy(pma, n_array, n_array_size * sizeof(float));

namespace glasssix
{
	public ref class ipbbox
	{
	private:
		int device_;
		ipbbox_net* ipbbox_net_;
	public:
		ipbbox(int device)
		{
			device_ = device;
			ipbbox_net_ = new ipbbox_net(device_);
		}

		!ipbbox()
		{
			delete ipbbox_net_;
			ipbbox_net_ = nullptr;
		}

		~ipbbox()
		{
			this->!ipbbox();
		}

		array<array<float>^>^ ExtractBitmapOutputs(array<Bitmap^>^ imgDatas, int DeviceId)
		{
			std::shared_ptr<tensor> tensor_data = nullptr;
			bitmaps2tensor(imgDatas, tensor_data);
			ipbbox_net_->Forward(tensor_data);
			auto outputs = gcnew array<array<float>^>(static_cast<int>(2));
			for (int i = 0; i < outputs->Length; i++)
			{
				if (i==0)
				{
					const float* intermediate = ipbbox_net_->get_fc2()->cpu_data();
					int output_size = ipbbox_net_->get_fc2()->count();
					MARSHAL_ARRAY(intermediate, output, output_size);
					outputs[i] = output;
				}
				if (i==1)
				{
					const float* intermediate = ipbbox_net_->get_fc3()->cpu_data();
					int output_size = ipbbox_net_->get_fc3()->count();
					MARSHAL_ARRAY(intermediate, output, output_size);
					outputs[i] = output;
				}
			}
			return outputs;
		}

	private:
		static void bitmaps2tensor(array<Bitmap^>^ bitmaps, std::shared_ptr<tensor>& tensor_data)
		{
			int num = bitmaps->Length;
			if (num <= 0)
			{
				return;
			}
			int channel = ipbbox_net::get_input_channel();
			float mean[] = { 104.0f, 117.0f, 124.0f };
			float scale = 0.0078125f;
			int width = ipbbox_net::get_input_width();
			int height = ipbbox_net::get_input_height();
			int n_offset = channel * height * width;
			int c_offset = height * width;

			Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);
			tensor_data.reset(new tensor(std::vector<int>{num, channel, height, width}, -1));
			float* float_data = tensor_data->mutable_cpu_data();

			for (int n = 0; n < num; n++)
			{
				Bitmap^ resize_bitmap;
				if (width == bitmaps[n]->Width && height == bitmaps[n]->Height)
				{
					// no other situations, only channel == 3 for unicorn net
					resize_bitmap = bitmaps[n]->Clone(rc, PixelFormat::Format24bppRgb);
				}
				else
				{
					resize_bitmap = gcnew Bitmap((Image ^)bitmaps[n], width, height);
					resize_bitmap = resize_bitmap->Clone(rc, PixelFormat::Format24bppRgb);
				}
				// get image data block
				BitmapData ^bmpData = resize_bitmap->LockBits(rc, ImageLockMode::ReadOnly, resize_bitmap->PixelFormat);
				pin_ptr<unsigned char> bmpBuffer = (unsigned char *)bmpData->Scan0.ToPointer();

				for (int c = 0; c < channel; ++c)
				{
					for (int h = 0; h < height; ++h)
					{
						int line_offset = h * bmpData->Stride + c;
						for (int w = 0; w < width; ++w)
						{
							float_data[n*n_offset + c*c_offset + h*width + w] =
								(static_cast<float>(bmpBuffer[line_offset + w * channel]) - mean[c]) * scale;
						}
					}
				}
				resize_bitmap->UnlockBits(bmpData);
			}
		}
	};

	public ref class ipts
	{
	private:
		int device_;
		ipts_net* ipts_net_;
	public:
		ipts(int device)
		{
			device_ = device;
			ipts_net_ = new ipts_net(device_);
		}

		!ipts()
		{
			delete ipts_net_;
			ipts_net_ = nullptr;
		}

		~ipts()
		{
			this->!ipts();
		}

		array<float>^ ExtractBitmapOutputs(array<Bitmap^>^ imgDatas, int DeviceId)
		{
			std::shared_ptr<tensor> tensor_data = nullptr;
			bitmaps2tensor(imgDatas, tensor_data);
			ipts_net_->Forward(tensor_data);
			const float* intermediate = ipts_net_->get_fc2()->cpu_data();
			int output_size = ipts_net_->get_fc2()->count();
			MARSHAL_ARRAY(intermediate, outputs, output_size);
			return outputs;
		}

	private:
		static void bitmaps2tensor(array<Bitmap^>^ bitmaps, std::shared_ptr<tensor>& tensor_data)
		{
			int num = bitmaps->Length;
			if (num <= 0)
			{
				return;
			}
			int channel = ipbbox_net::get_input_channel();
			float mean[] = { 104.0f, 117.0f, 124.0f };
			float scale = 0.0078125f;
			int width = ipbbox_net::get_input_width();
			int height = ipbbox_net::get_input_height();
			int n_offset = channel * height * width;
			int c_offset = height * width;

			Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);
			tensor_data.reset(new tensor(std::vector<int>{num, channel, height, width}, -1));
			float* float_data = tensor_data->mutable_cpu_data();

			for (int n = 0; n < num; n++)
			{
				Bitmap^ resize_bitmap;
				if (width == bitmaps[n]->Width && height == bitmaps[n]->Height)
				{
					// no other situations, only channel == 3 for unicorn net
					resize_bitmap = bitmaps[n]->Clone(rc, PixelFormat::Format24bppRgb);
				}
				else
				{
					resize_bitmap = gcnew Bitmap((Image ^)bitmaps[n], width, height);
					resize_bitmap = resize_bitmap->Clone(rc, PixelFormat::Format24bppRgb);
				}
				// get image data block
				BitmapData ^bmpData = resize_bitmap->LockBits(rc, ImageLockMode::ReadOnly, resize_bitmap->PixelFormat);
				pin_ptr<unsigned char> bmpBuffer = (unsigned char *)bmpData->Scan0.ToPointer();

				for (int c = 0; c < channel; ++c)
				{
					for (int h = 0; h < height; ++h)
					{
						int line_offset = h * bmpData->Stride + c;
						for (int w = 0; w < width; ++w)
						{
							float_data[n*n_offset + c*c_offset + h*width + w] =
								(static_cast<float>(bmpBuffer[line_offset + w * channel]) - mean[c]) * scale;
						}
					}
				}
				resize_bitmap->UnlockBits(bmpData);
			}
		}

	};
}