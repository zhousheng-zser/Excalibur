#include <msclr\marshal_cppstd.h>
#include "unicorn_net.hpp"

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

#define MODEL_VERSION "9722"

namespace glasssix
{
	namespace excalibur
	{
		namespace cassius
		{
			public ref class Unicorn
			{
			private:
				int device_;
				unicorn_net* net_;
			public:
				Unicorn(int device)
				{
					device_ = device;
					net_ = new unicorn_net(device_);
				}

				!Unicorn()
				{
					delete net_;
					net_ = nullptr;
				}

				~Unicorn()
				{
					this->!Unicorn();
				}

				static String^ GetModelVersion()
				{
					return gcnew String(MODEL_VERSION);
				}

				String^ version()
				{
					return gcnew String("1.0.0");
				}

				String^ description()
				{
					return gcnew String("Baseline version, with quality score supoort with cudnn.");
				}

				array<float>^ ExtractBitmapOutputs(array<Bitmap^>^ imgDatas)
				{
					std::shared_ptr<tensor<float>> tensor_data = nullptr;
					bitmaps2tensor(imgDatas, tensor_data);
					net_->Forward(tensor_data);
					const float* intermediate = net_->get_pool5()->cpu_data();
					int output_size = net_->get_pool5()->count();
					MARSHAL_ARRAY(intermediate, outputs, output_size);
					return outputs;
				}

				array<float>^ GetQualityScores()
				{
					std::vector<float> qs = net_->get_quality_score();
					auto output = gcnew array<float>(qs.size());
					for (int i = 0; i < qs.size(); i++)
					{
						output[i] = qs[i];
					}
					return output;
				}

				static float CosineDistanceProb(array<float>^ feature1, array<float>^ feature2)
				{
					float output = 0;
					if (feature1->Length != feature2->Length)
					{
						output = -1;
					}
					else
					{
						output = innerproduct(feature1, feature2)
							/ Math::Sqrt(innerproduct(feature1, feature1)*innerproduct(feature2, feature2));
					}
					return output;
				}

				static float EuclideanDistanceProb(array<float>^ feature1, array<float>^ feature2)
				{
					float output = 0;
					if (feature1->Length != feature2->Length)
					{
						output = -1;
					}
					else
					{
						for (int i = 0; i < feature1->Length; i++)
						{
							output += (feature1[i] - feature2[i])*(feature1[i] - feature2[i]);
						}
					}
					return Math::Sqrt(output*1.0f);
				}

			private:
				static void bitmaps2tensor(array<Bitmap^>^ bitmaps, std::shared_ptr<tensor<float>>& tensor_data)
				{
					int num = bitmaps->Length;
					if (num <= 0)
					{
						return;
					}
					int channel = unicorn_net::get_input_channel();
					float mean[] = { 104.0f, 117.0f, 124.0f };
					float scale = 0.0078125f;
					int width = unicorn_net::get_input_width();
					int height = unicorn_net::get_input_height();
					int n_offset = channel * height * width;
					int c_offset = height * width;

					Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);
					tensor_data.reset(new tensor<float>(std::vector<int>{num, channel, height, width}, -1));
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

				static double innerproduct(array<float>^ feature1, array<float>^ feature2)
				{
					double output = 0;
					for (int i = 0; i < feature1->Length; i++)
					{
						output += feature1[i] * feature2[i];
					}
					return output;
				}
			};
		}
	}
	
}