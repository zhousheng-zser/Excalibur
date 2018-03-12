#include <msclr\marshal_cppstd.h>
#include "alcnn.hpp"

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
	namespace excalibur
	{
		namespace longinus
		{
			public ref class Banshee
			{
			private:
				int device_;
				alcnn* alcnn_net_;
			public:
				Banshee(int device)
				{
					device_ = device;
					alcnn_net_ = new alcnn(device_);
				}
				!Banshee()
				{
					delete alcnn_net_;
					alcnn_net_ = nullptr;
				}

				~Banshee()
				{
					this->!Banshee();
				}

				array<array<float>^>^ ExtractBitmapOutputs_IPBbox(array<Bitmap^>^ imgDatas)
				{
					std::shared_ptr<tensor> tensor_data = nullptr;
					bitmaps2tensor(imgDatas, tensor_data);
					alcnn_net_->Forward_IPBbox(tensor_data);
					auto outputs = gcnew array<array<float>^>(static_cast<int>(2));
					for (int i = 0; i < outputs->Length; i++)
					{
						if (i == 0)
						{
							const float* intermediate = alcnn_net_->get_IPBbox_fc2_data();
							int output_size = alcnn_net_->get_IPBbox_fc2_count();
							MARSHAL_ARRAY(intermediate, output, output_size);
							outputs[i] = output;
						}
						if (i == 1)
						{
							const float* intermediate = alcnn_net_->get_IPBbox_fc3_data();
							int output_size = alcnn_net_->get_IPBbox_fc3_count();
							MARSHAL_ARRAY(intermediate, output, output_size);
							outputs[i] = output;
						}
					}
					return outputs;
				}

				array<float>^ ExtractBitmapOutputs_IPTs(array<Bitmap^>^ imgDatas)
				{
					std::shared_ptr<tensor> tensor_data = nullptr;
					bitmaps2tensor(imgDatas, tensor_data);
					alcnn_net_->Forward_IPTs(tensor_data);
					const float* intermediate = alcnn_net_->get_IPTs_fc2_data();
					int output_size = alcnn_net_->get_IPTs_fc2_count();
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
					int channel = 3;// ipbbox_net::get_input_channel();
					float mean[] = { 104.0f, 117.0f, 124.0f };
					float scale = 0.0078125f;
					int width = 60;// ipbbox_net::get_input_width();
					int height = 60;// ipbbox_net::get_input_height();
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
	}
}