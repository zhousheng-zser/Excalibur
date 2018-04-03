// This is the main DLL file.

#include "stdafx.h"

#include "FastRomancia.hpp"

namespace glasssix
{
	namespace aroundight
	{
		namespace romancia
		{

			FastFace::FastFace(int device)
			{
				nw_ = new npd_wrapper(device);
			}

			FastFace::~FastFace()
			{
				this->!FastFace();
			}

			FastFace::!FastFace()
			{
				delete nw_;
				nw_ = NULL;
			}

			unsigned char* FastFace::Bitmap2Gray(System::Drawing::Bitmap^ bmp, int& stride)
			{
				//Console::WriteLine(bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed);
				System::Drawing::Imaging::BitmapData^ bmpd;
				if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed) {

					//Console::WriteLine("before malloc data!");
					bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
					//Console::WriteLine("after malloc data!");
					stride = bmpd->Stride;
					return (unsigned char*)bmpd->Scan0.ToPointer();
				}

				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
				stride = bmp->Width + (4 - bmp->Width % 4) % 4;
				unsigned char* res = new unsigned char[stride * bmp->Height];
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer(),
					*b, *g, *r;
				int offset = 0;
				for (int y = 0; y < bmp->Height; ++y)
				{
					b = pBmp + offset + 0, g = pBmp + offset + 1, r = pBmp + offset + 2;
					for (int x = 0; x < bmpd->Width; ++x) {
						res[y * stride + x] = (unsigned char)((float)*r * 0.3f + (float)*g * 0.59f + (float)*b * 0.11f);
						b += 3, g += 3, r += 3;
					}
					offset += bmpd->Stride;
				}
				bmp->UnlockBits(bmpd);
				return res;
			}

			List<FaceInfo>^ FastFace::Facedetect_Frontal_Reinforce(System::Drawing::Bitmap^ Oribmp, int min_size, float scale)
			{
				int stride;
				unsigned char * buf = Bitmap2Gray(Oribmp, stride);
				int n = nw_->facedetect_npd(buf, Oribmp->Width, Oribmp->Height, min_size);
				List<FaceInfo>^ output = gcnew List<FaceInfo>();
				auto X = nw_->get_x();
				auto Y = nw_->get_y();
				auto S = nw_->get_size();
				auto scores = nw_->get_score();
				for (int i = 0; i < n; i++)
				{
					output->Add(FaceInfo(i, 0, scores[i], System::Drawing::Rectangle(X[i], Y[i], S[i], S[i])));
				}
				delete[] buf;
				return output;
			}
		}
	}
}