#include "UTensor.hpp"
using namespace excalibur;

namespace glasssix
{
	namespace gilgamesh
	{
		UTensor::UTensor()
		{
			num = 0;
			channel = 0;
			width = 0;
			height = 0;
			device = -1;
			data = new utensor();
		}

		UTensor::!UTensor()
		{
			delete data;
			data = nullptr;
		}

		UTensor::~UTensor()
		{
			this->!UTensor();
		}

		UTensor::UTensor(const UTensor %t)
		{
			num = t.num;
			channel = t.channel;
			width = t.width;
			height = t.height;
			device = t.device;
			data = &(t.data->clone());
		}

		UTensor::UTensor(int num, int channel, int height, int width, int device)
		{
			this->num = num;
			this->channel = channel;
			this->width = width;
			this->height = height;
			this->device = device;
			data = new utensor(std::vector<int>
			{this->num, this->channel, this->height, this->width}, this->device);
		}

		UTensor::UTensor(int channel, int height, int width, int device)
		{
			this->num = 1;
			this->channel = channel;
			this->width = width;
			this->height = height;
			this->device = device;
			data = new utensor(std::vector<int>
			{this->num, this->channel, this->height, this->width}, this->device);
		}

		UTensor::UTensor(int height, int width, int device)
		{
			this->num = 1;
			this->channel = 1;
			this->width = width;
			this->height = height;
			this->device = device;
			data = new utensor(std::vector<int>
			{this->num, this->channel, this->height, this->width}, this->device);
		}

		UTensor::UTensor(int size, int device)
		{
			this->num = 1;
			this->channel = size;
			this->width = 1;
			this->height = 1;
			this->device = device;
			data = new utensor(std::vector<int>
			{this->num, this->channel, this->height, this->width}, this->device);
		}

		UTensor::UTensor(Bitmap^ bmp, int device)
		{
			this->num = 1;
			this->width = bmp->Width;
			this->height = bmp->Height;
			this->device = device;
			System::Drawing::Imaging::BitmapData^ bmpd;
			//gray8
			if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)
			{
				this->channel = 1;
				data = new utensor(std::vector<int>
				{this->num, this->channel, this->height, this->width}, this->device);
				unsigned char* dst_data = data->getdata()->mutable_cpu_data();
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				int stride = (width * channel + 3) & -4;
				for (int h = 0; h < this->height; h++)
				{
					int offset = h * this->width;
					int h_stride = h * stride;
					for (int w = 0; w < this->width; w++)
					{
						dst_data[offset + w] = (unsigned char)pBmp[h_stride + w];
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//rgb24
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
			{
				this->channel = 3;
				data = new utensor(std::vector<int>
				{this->num, this->channel, this->height, this->width}, this->device);
				unsigned char* dst_data = data->getdata()->mutable_cpu_data();
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				int stride = (width * channel + 3) & -4;
				int offset = this->width * this->height;
				for (int c = 0; c < channel; c++)
				{
					int c_offset = offset * c;
					for (int h = 0; h < this->height; h++)
					{
						int sub_offset = h * this->width;
						int h_stride = h * stride;
						for (int w = 0; w < this->width; w++)
						{
							dst_data[c_offset + sub_offset + w]
								= (unsigned char)pBmp[h_stride + w * 3 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//rgba32
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)
			{
				this->channel = 4;
				data = new utensor(std::vector<int>
				{this->num, this->channel, this->height, this->width}, this->device);
				unsigned char* dst_data = data->getdata()->mutable_cpu_data();
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				for (int c = 0; c < this->channel; c++)
				{
					int offset = this->width * this->height * c;
					for (int h = 0; h < this->height; h++)
					{
						int sub_offset = h * this->width;
						for (int w = 0; w < this->width; w++)
						{
							dst_data[offset + sub_offset + w] =
								(unsigned char)pBmp[(sub_offset + w) * 4 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			// else, convert to rgb24
			else
			{
				this->channel = 3;
				data = new utensor(std::vector<int>
				{this->num, this->channel, this->height, this->width}, this->device);
				unsigned char* dst_data = data->getdata()->mutable_cpu_data();
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				int stride = (width * channel + 3) & -4;
				int offset = this->width * this->height;
				for (int c = 0; c < channel; c++)
				{
					int c_offset = offset * c;
					for (int h = 0; h < this->height; h++)
					{
						int sub_offset = h * this->width;
						int h_stride = h * stride;
						for (int w = 0; w < this->width; w++)
						{
							dst_data[c_offset + sub_offset + w]
								= (unsigned char)pBmp[h_stride + w * 3 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
		}

		UTensor::UTensor(String^ path, int device)
		{
			Bitmap^ bmp = gcnew Bitmap(path);
			this->num = 1;
			this->width = bmp->Width;
			this->height = bmp->Height;
			this->device = device;
			System::Drawing::Imaging::BitmapData^ bmpd;
			//gray8
			if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)
			{
				this->channel = 1;
				data = new utensor(std::vector<int>
				{this->num, this->channel, this->height, this->width}, this->device);
				unsigned char* dst_data = data->getdata()->mutable_cpu_data();
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				int stride = (width * channel + 3) & -4;
				for (int h = 0; h < this->height; h++)
				{
					int offset = h * this->width;
					int h_stride = h * stride;
					for (int w = 0; w < this->width; w++)
					{
						dst_data[offset + w] = (unsigned char)pBmp[h_stride + w];
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//rgb24
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
			{
				this->channel = 3;
				data = new utensor(std::vector<int>
				{this->num, this->channel, this->height, this->width}, this->device);
				unsigned char* dst_data = data->getdata()->mutable_cpu_data();
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				int stride = (width * channel + 3) & -4;
				int offset = this->width * this->height;
				for (int c = 0; c < channel; c++)
				{
					int c_offset = offset * c;
					for (int h = 0; h < this->height; h++)
					{
						int sub_offset = h * this->width;
						int h_stride = h * stride;
						for (int w = 0; w < this->width; w++)
						{
							dst_data[c_offset + sub_offset + w]
								= (unsigned char)pBmp[h_stride + w * 3 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//rgba32
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)
			{
				this->channel = 4;
				data = new utensor(std::vector<int>
				{this->num, this->channel, this->height, this->width}, this->device);
				unsigned char* dst_data = data->getdata()->mutable_cpu_data();
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				for (int c = 0; c < this->channel; c++)
				{
					int offset = this->width * this->height * c;
					for (int h = 0; h < this->height; h++)
					{
						int sub_offset = h * this->width;
						for (int w = 0; w < this->width; w++)
						{
							dst_data[offset + sub_offset + w] =
								(unsigned char)pBmp[(sub_offset + w) * 4 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			// else, convert to rgb24
			else
			{
				this->channel = 3;
				data = new utensor(std::vector<int>
				{this->num, this->channel, this->height, this->width}, this->device);
				unsigned char* dst_data = data->getdata()->mutable_cpu_data();
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				int stride = (width * channel + 3) & -4;
				int offset = this->width * this->height;
				for (int c = 0; c < channel; c++)
				{
					int c_offset = offset * c;
					for (int h = 0; h < this->height; h++)
					{
						int sub_offset = h * this->width;
						int h_stride = h * stride;
						for (int w = 0; w < this->width; w++)
						{
							dst_data[c_offset + sub_offset + w]
								= (unsigned char)pBmp[h_stride + w * 3 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			delete bmp;
		}

		Bitmap^ UTensor::ToBitmap()
		{
			if (this->channel > 4)
			{
				// Un-support convertion
				return gcnew System::Drawing::Bitmap(0, 0);
			}
			else if (this->channel == 4)
			{
				// rgba32, the data pointer is aligned by itself
				unsigned char* dst_data = new unsigned char[width * height * 4];
				const unsigned char* src_data = data->getdata()->cpu_data();
				for (int c = 0; c < channel; c++)
				{
					int offset = this->width * this->height * c;
					for (int h = 0; h < height; h++)
					{
						int sub_offset = h * this->width;
						for (int w = 0; w < width; w++)
						{
							dst_data[(sub_offset + w) * 4 + c] =
								(unsigned char)src_data[offset + sub_offset + w];
						}
					}
				}
				System::IntPtr ptr = (System::IntPtr)dst_data;
				System::Drawing::Bitmap^ bmp = gcnew System::Drawing::Bitmap(width, height,
					width * 4,
					System::Drawing::Imaging::PixelFormat::Format32bppArgb,
					ptr);
				return bmp;
			}
			else if (this->channel == 3)
			{
				System::Drawing::Bitmap^ bmp =
					gcnew Bitmap(width, height, PixelFormat::Format24bppRgb);
				BitmapData^ bmpdata = bmp->LockBits(
					System::Drawing::Rectangle(Point::Empty, Size(width, height)),
					ImageLockMode::WriteOnly,
					PixelFormat::Format24bppRgb);
				unsigned char* dst_data = (unsigned char*)bmpdata->Scan0.ToPointer();
				const unsigned char* src_data = data->getdata()->cpu_data();
				int* offset = new int[channel];
				for (int c = 0; c < channel; c++)
				{
					offset[c] = c * this->width * this->height;
				}
				int stride = width * channel;
				int aligned_stride = (width * channel + 3) & -4;
				int dst_aligned_offset = 0;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * this->width;
					int aligned_stride_h = aligned_stride * h;
					for (int w = 0; w < width; w++)
					{
						for (int c = 0; c < channel; c++)
						{
							dst_data[aligned_stride_h + w * channel + c] =
								(unsigned char)src_data[offset[c] + sub_offset + w];
						}
					}
					int stride_offset_counter = stride;
					//fill the offset
					while (stride_offset_counter % 4 != 0)
					{
						dst_aligned_offset += 1;
						dst_data[(h + 1) * stride + dst_aligned_offset] = 0;
						stride_offset_counter += 1;
					}
				}
				bmp->UnlockBits(bmpdata);
				return bmp;
			}
			else if (this->channel == 1)
			{
				System::Drawing::Bitmap^ bmp =
					gcnew Bitmap(width, height, PixelFormat::Format8bppIndexed);
				BitmapData^ bmpdata = bmp->LockBits(
					System::Drawing::Rectangle(Point::Empty, Size(width, height)),
					ImageLockMode::WriteOnly,
					PixelFormat::Format8bppIndexed);
				unsigned char* dst_data = (unsigned char*)bmpdata->Scan0.ToPointer();
				const unsigned char* src_data = data->getdata()->cpu_data();
				int* offset = new int[3];
				for (int c = 0; c < 3; c++)
				{
					offset[c] = 0 * this->width * this->height;
				}
				int stride = width * 1;
				int aligned_stride = (width * 1 + 3) & -4;
				int dst_aligned_offset = 0;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * this->width;
					int aligned_stride_h = aligned_stride * h;
					for (int w = 0; w < width; w++)
					{
						for (int c = 0; c < 1; c++)
						{
							dst_data[aligned_stride_h + w * 1 + c] =
								(unsigned char)src_data[offset[c] + sub_offset + w];
						}
					}
					int stride_offset_counter = stride;
					//fill the offset
					while (stride_offset_counter % 4 != 0)
					{
						dst_aligned_offset += 1;
						dst_data[(h + 1) * stride + dst_aligned_offset] = 0;
						stride_offset_counter += 1;
					}
				}
				bmp->UnlockBits(bmpdata);
				return bmp;
			}
			else
			{
				// channel == 2, do not support now
				return gcnew System::Drawing::Bitmap(0, 0);
			}
		}

		void UTensor::Save(String^ path, ImageEncodingType type)
		{
			if (this->Channel == 1)
			{
				// There is an unknown phenomenon in .NET GDI+(1.0) that\ 
				// Bitmap 'save' mathod will ignored pixelformat settings.
				// Fix it in next version without GDI+ solution.
				// Now, return directly.
				// https://stackoverflow.com/questions/4679827/c-sharp-why-bitmap-save-ignores-pixelformat-of-bitmap
				return;
			}
			if (this->data->empty())
			{
				LOG(ERROR) << "Try to save empty UTensor.";
				return;
			}
			Bitmap^ bmp = ToBitmap();

			switch (type)
			{
			case glasssix::gilgamesh::ImageEncodingType::Native:
				NOT_IMPLEMENTED;
				break;
			case glasssix::gilgamesh::ImageEncodingType::Bmp:
				bmp->Save(path, ImageFormat::Bmp);
				break;
			case glasssix::gilgamesh::ImageEncodingType::Png:
				bmp->Save(path, ImageFormat::Png);
				break;
			case glasssix::gilgamesh::ImageEncodingType::Jpeg:
				bmp->Save(path, ImageFormat::Jpeg);
				break;
			default:
				LOG(ERROR) << "Un-known Encoding type.";
				break;
			}
		}
	}
}