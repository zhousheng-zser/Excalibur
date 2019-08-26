#include "cassiunia.hpp"

#define MARSHAL_ARRAY(n_array, m_array, n_array_size) \
  auto m_array = gcnew array<float>(n_array_size); \
  pin_ptr<float> pma = &m_array[0]; \
  memcpy(pma, n_array, n_array_size * sizeof(float));

namespace glasssix
{
	namespace cassius
	{
		Cassiunia::Cassiunia(int device)
		{
			this->device_ = device;
			cassius_wrapper = new CassiusFeature(device_);
		}

		Cassiunia::~Cassiunia()
		{
			this->!Cassiunia();
		}

		Cassiunia::!Cassiunia()
		{
			delete cassius_wrapper;
			cassius_wrapper = nullptr;
		}

		unsigned char* Cassiunia::Bitmaps2RGBs(array<Bitmap^>^ bmps)
		{
			int num = bmps->Length;
			if (num <= 0)
			{
				return nullptr;
			}
			int channel = 3;
			int width = 128;
			int height = 128;
			int n_offset = channel * height * width;
			int c_offset = height * width;
			unsigned char* dst_data = new unsigned char[num * n_offset];
			//Convert to RGB24(NCHW)
			for (size_t i = 0; i < num; i++)
			{
				unsigned char* dst_data_3c = new unsigned char[n_offset];
				if (bmps[i]->Width != 128 || bmps[i]->Height != 128)
				{
					memset(dst_data_3c, 0, n_offset * sizeof(unsigned char));
				}
				else
				{
					System::Drawing::Imaging::BitmapData^ bmpd;
					bmpd = bmps[i]->LockBits(System::Drawing::Rectangle(0, 0, bmps[i]->Width, bmps[i]->Height),
						System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
					unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
					int stride = (width * 3 + 3) & -4;
					int offset = width * height;
					for (int c = 0; c < 3; c++)
					{
						int c_offset = offset * c;
						for (int h = 0; h < height; h++)
						{
							int sub_offset = h * width;
							int h_stride = h * stride;
							for (int w = 0; w < width; w++)
							{
								dst_data_3c[c_offset + sub_offset + w]
									= (unsigned char)pBmp[h_stride + w * 3 + c];
							}
						}
					}
					bmps[i]->UnlockBits(bmpd);
				}
				memcpy(dst_data + i * n_offset, dst_data_3c, n_offset * sizeof(unsigned char));
				delete dst_data_3c;
			}
			return dst_data;
		}

		array<float>^ Cassiunia::ExtractBitmapOutputs(array<Bitmap^>^ imgDatas)
		{
			if (imgDatas->Length <= 0)
			{
				return gcnew array<float>(0);
			}
			auto data = Bitmaps2RGBs(imgDatas);
			if (!data)
			{
				return gcnew array<float>(0);
			}
			auto res = cassius_wrapper->Forward(data, imgDatas->Length);
			auto m_array = gcnew array<float>(512 * imgDatas->Length);
			pin_ptr<float> pma = &m_array[0];
			for (size_t i = 0; i < imgDatas->Length; i++)
			{
				memcpy(pma + i * 512, res[i].data(), 512 * sizeof(float));
			}	

			delete[] data;
			return m_array;
		}
	}
}