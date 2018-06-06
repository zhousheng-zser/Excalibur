#include "Tensor.hpp"
using namespace excalibur;

namespace glasssix
{
	namespace gilgamesh
	{
		Tensor::Tensor()
		{
			num = 0;
			channel = 0;
			width = 0;
			height = 0;
			device = -1;
			data = new ftensor();
		}

		Tensor::!Tensor()
		{
			delete data;
			data = nullptr;
		}

		Tensor::~Tensor()
		{
			this->!Tensor();
		}

		Tensor::Tensor(const Tensor %t)
		{
			num = t.num;
			channel = t.channel;
			width = t.width;
			height = t.height;
			device = t.device;
			data = &(t.data->clone());
		}

		Tensor::Tensor(int num, int channel, int height, int width, int device)
		{
			this->num = num;
			this->channel = channel;
			this->width = width;
			this->height = height;
			this->device = device;
			data = new ftensor(std::vector<int>
			{this->num, this->channel, this->height, this->width}, this->device);
		}

		Tensor::Tensor(int channel, int height, int width, int device)
		{
			this->num = 1;
			this->channel = channel;
			this->width = width;
			this->height = height;
			this->device = device;
			data = new ftensor(std::vector<int>
			{this->num, this->channel, this->height, this->width}, this->device);
		}

		Tensor::Tensor(int height, int width, int device)
		{
			this->num = 1;
			this->channel = 1;
			this->width = width;
			this->height = height;
			this->device = device;
			data = new ftensor(std::vector<int>
			{this->num, this->channel, this->height, this->width}, this->device);
		}

		Tensor::Tensor(int size, int device)
		{
			this->num = 1;
			this->channel = size;
			this->width = 1;
			this->height = 1;
			this->device = device;
			data = new ftensor(std::vector<int>
			{this->num, this->channel, this->height, this->width}, this->device);
		}

		Tensor::Tensor(Bitmap^ bmp, int device)
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
				data = new ftensor(std::vector<int>
				{this->num, this->channel, this->width, this->height}, this->device);
				float* dst_data = data->getdata()->mutable_cpu_data();
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
						dst_data[offset + w] = (float)pBmp[h_stride + w];
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//rgb24
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
			{
				this->channel = 3;
				data = new ftensor(std::vector<int>
				{this->num, this->channel, this->width, this->height}, this->device);
				float* dst_data = data->getdata()->mutable_cpu_data();
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
								= (float)pBmp[h_stride + w * 3 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//rgba32
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)
			{
				this->channel = 4;
				data = new ftensor(std::vector<int>
				{this->num, this->channel, this->width, this->height}, this->device);
				float* dst_data = data->getdata()->mutable_cpu_data();
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
								(float)pBmp[(sub_offset + w) * 4 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			// else, convert to rgb24
			else
			{
				this->channel = 3;
				data = new ftensor(std::vector<int>
				{this->num, this->channel, this->width, this->height}, this->device);
				float* dst_data = data->getdata()->mutable_cpu_data();
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
								= (float)pBmp[h_stride + w * 3 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
		}

		Tensor::Tensor(String^ path, int device)
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
				data = new ftensor(std::vector<int>
				{this->num, this->channel, this->width, this->height}, this->device);
				float* dst_data = data->getdata()->mutable_cpu_data();
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
						dst_data[offset + w] = (float)pBmp[h_stride + w];
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//rgb24
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
			{
				this->channel = 3;
				data = new ftensor(std::vector<int>
				{this->num, this->channel, this->width, this->height}, this->device);
				float* dst_data = data->getdata()->mutable_cpu_data();
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
								= (float)pBmp[h_stride + w * 3 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//rgba32
			else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)
			{
				this->channel = 4;
				data = new ftensor(std::vector<int>
				{this->num, this->channel, this->width, this->height}, this->device);
				float* dst_data = data->getdata()->mutable_cpu_data();
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
								(float)pBmp[(sub_offset + w) * 4 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			// else, convert to rgb24
			else
			{
				this->channel = 3;
				data = new ftensor(std::vector<int>
				{this->num, this->channel, this->width, this->height}, this->device);
				float* dst_data = data->getdata()->mutable_cpu_data();
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
								= (float)pBmp[h_stride + w * 3 + c];
						}
					}
				}
				bmp->UnlockBits(bmpd);
			}
			delete bmp;
		}

		Bitmap^ Tensor::ToBitmap()
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
				const float* src_data = data->getdata()->cpu_data();
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
			else if (this->channel == 3 || this->channel == 1)
			{
				// rgb24 or gray8, alignment check is neccessary
				// pointer aligned to 4's multiple
				System::Drawing::Bitmap^ bmp;
				const float* src_data = data->getdata()->cpu_data();
				int* offset = new int[channel];
				for (int c = 0; c < channel; c++)
				{
					offset[c] = c * this->width * this->height;
				}
				int stride = width * channel;
				if (stride % 4 != 0)
				{
					unsigned char* dst_data = new unsigned char[((width * channel + 3) & -4) * height];
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
						do
						{
							dst_aligned_offset += 1;
							dst_data[(h + 1) * stride + dst_aligned_offset] = 0;
							stride_offset_counter += 1;
						} while (stride_offset_counter % 4 != 0);
					}
					System::IntPtr ptr = (System::IntPtr)dst_data;
					//aligned_stride = (width * channel + 3) & -4;
					if (channel == 1)
					{
						// gray 8
						bmp = gcnew System::Drawing::Bitmap(width, height,
							aligned_stride,
							System::Drawing::Imaging::PixelFormat::Format8bppIndexed,
							ptr);
					}
					else
					{
						// rgb24
						bmp = gcnew System::Drawing::Bitmap(width, height,
							aligned_stride,
							System::Drawing::Imaging::PixelFormat::Format24bppRgb,
							ptr);
					}
				}
				else
				{
					unsigned char* dst_data = new unsigned char[width * channel * height];
					for (int h = 0; h < height; h++)
					{
						int sub_offset = h * this->width;
						for (int w = 0; w < width; w++)
						{
							for (int c = 0; c < channel; c++)
							{
								dst_data[(sub_offset + w) * channel + c] =
									(unsigned char)src_data[offset[c] + sub_offset + w];
							}
						}
					}
					System::IntPtr ptr = (System::IntPtr)dst_data;
					if (channel == 1)
					{
						// gray 8
						bmp = gcnew System::Drawing::Bitmap(width, height,
							stride,
							System::Drawing::Imaging::PixelFormat::Format8bppIndexed,
							ptr);
					}
					else
					{
						// rgb24
						bmp = gcnew System::Drawing::Bitmap(width, height,
							stride,
							System::Drawing::Imaging::PixelFormat::Format24bppRgb,
							ptr);
					}
				}
				delete offset;
				return bmp;
			}
			else
			{
				// channel == 2, do not support now
				return gcnew System::Drawing::Bitmap(0, 0);
			}
		}

		void Tensor::Save(String^ path, ImageEncodingType type)
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
				LOG(ERROR) << "Try to save empty tensor.";
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