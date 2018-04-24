#include "romancia.hpp"
#include <facedetect-dll.h>

namespace glasssix
{
	namespace aroundight
	{
		namespace romancia
		{

			FastFace::FastFace(int device)
			{
				mw_ = new mtcnn_warpper(device);
				nw_ = new npd_wrapper(device);
			}

			FastFace::~FastFace()
			{
				this->!FastFace();
			}

			FastFace::!FastFace()
			{
				delete mw_;
				mw_ = NULL;
				delete nw_;
				nw_ = NULL;
			}

			unsigned char* FastFace::Bitmap2RGB(System::Drawing::Bitmap^ bmp)
			{
				int stride;
				unsigned char* res;
				System::Drawing::Imaging::BitmapData^ bmpd;
				if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed) 
				{
					stride = bmpd->Stride;
					bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
					return (unsigned char*)bmpd->Scan0.ToPointer();
				}
				else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
				{
					bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
					stride = bmp->Width;
					res = new unsigned char[stride * bmp->Height * 3];
					unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer(),
						*b, *g, *r;
					for (int offset = 0, y = 0; y < bmp->Height; ++y, offset += bmpd->Stride) 
					{
						b = pBmp + offset + 0, g = pBmp + offset + 1, r = pBmp + offset + 2;
						for (int x = 0; x < bmpd->Width; ++x, b += 3, g += 3, r += 3) 
						{
							res[(y * stride + x) * 3] = *b;
							res[(y * stride + x) * 3 + 1] = *g;
							res[(y * stride + x) * 3 + 2] = *r;
						}
					}
					bmp->UnlockBits(bmpd);
				}
				else
				{
					res = nullptr;//logical complication
				}
				return res;
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
				stride = bmp->Width;
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

			List<FaceInfo>^ FastFace::GetResultsList(int * pResults)
			{
				List<FaceInfo>^ output = gcnew List<FaceInfo>();
				if (!pResults)
				{
					return output;
				}
				for (int i = 0; i < (pResults ? *pResults : 0); i++)
				{
					short * p = ((short*)(pResults + 1)) + 142 * i;
					int x = p[0];
					int y = p[1];
					int w = p[2];
					int h = p[3];
					float neighbors = (float)p[4];
					int angle = p[5];
					array<int>^ landmark = gcnew array<int>(136);
					if (dolandmark_)
					{
						for (int j = 0; j < 68; j++)
						{
							landmark[2 * j] = (int)p[6 + 2 * j];
							landmark[2 * j + 1] = (int)p[6 + 2 * j + 1];
						}
						output->Add(FaceInfo(i, angle, neighbors, System::Drawing::Rectangle(x, y, w, h), landmark));
					}
					else
					{
						output->Add(FaceInfo(i, angle, neighbors, System::Drawing::Rectangle(x, y, w, h)));
					}
				}
				return output;
			}

			List<FaceInfo>^ FastFace::Facedetect_Frontal(System::Drawing::Bitmap^ Oribmp, int min_size, float scale)
			{
				int stride;
				unsigned char * buf = Bitmap2Gray(Oribmp, stride);
				unsigned char * pBuffer = new unsigned char[0x20000];
				if (!pBuffer)
				{
					Console::WriteLine("Can not alloc buffer.");
					return gcnew List<FaceInfo>();
				}
				List<FaceInfo>^ output = GetResultsList(facedetect_frontal(pBuffer, buf, Oribmp->Width, Oribmp->Height, stride,
					scale, threshold_, min_size, 0, (int)dolandmark_));
				delete[] buf;
				delete[] pBuffer;
				return output;
			}

			List<FaceInfo>^ FastFace::Facedetect_Multiview(System::Drawing::Bitmap^ Oribmp, int min_size, float scale)
			{
				int stride;
				unsigned char * buf = Bitmap2Gray(Oribmp, stride);
				unsigned char * pBuffer = new unsigned char[0x20000];
				if (!pBuffer)
				{
					Console::WriteLine("Can not alloc buffer.");
					return gcnew List<FaceInfo>();
				}
				List<FaceInfo>^ output = GetResultsList(facedetect_multiview(pBuffer, buf, Oribmp->Width, Oribmp->Height, stride,
					scale, threshold_, min_size, 0, (int)dolandmark_));
				delete[] buf;
				delete[] pBuffer;
				return output;
			}

			List<FaceInfo>^ FastFace::Facedetect_Multiview_Reinforce(System::Drawing::Bitmap^ Oribmp, int min_size, float scale)
			{
				int stride;
				unsigned char * buf = Bitmap2Gray(Oribmp, stride);
				unsigned char * pBuffer = new unsigned char[0x20000];
				if (!pBuffer)
				{
					Console::WriteLine("Can not alloc buffer.");
					return gcnew List<FaceInfo>();
				}
				List<int>^ angle = gcnew List<int>();
				List<FaceInfo>^ output = GetResultsList(facedetect_multiview_reinforce(pBuffer, buf, Oribmp->Width, Oribmp->Height, stride,
					scale, threshold_, min_size, 0, (int)dolandmark_));
				delete[] buf;
				delete[] pBuffer;
				return output;
			}

			List<FaceInfo>^ FastFace::Facedetect_Frontal_Surveillance(System::Drawing::Bitmap^ Oribmp, int min_size, float scale)
			{
				int stride;
				unsigned char * buf = Bitmap2Gray(Oribmp, stride);
				unsigned char * pBuffer = new unsigned char[0x20000];

				if (!pBuffer)
				{
					Console::WriteLine("Can not alloc buffer.");
					return gcnew List<FaceInfo>();
				}
				List<int>^ angle = gcnew List<int>();
				List<FaceInfo>^ output = GetResultsList(facedetect_frontal_surveillance(pBuffer, buf, Oribmp->Width, Oribmp->Height, stride,
					scale, threshold_, min_size, 0, (int)dolandmark_));
				delete[] buf;
				delete[] pBuffer;
				return output;
			}
			
			List<FaceInfo>^ FastFace::Facedetect_Multiview_CNN(System::Drawing::Bitmap^ Oribmp, int min_size, float scale)
			{
				std::vector<FaceInfoX> infos = mw_->facedetect_mtcnn(Bitmap2RGB(Oribmp), Oribmp->Width, Oribmp->Height, min_size);
				List<FaceInfo>^ output = gcnew List<FaceInfo>();
				for (int i = 0; i < infos.size(); i++)
				{
					int x = (int)infos[i].bbox.xmin;
					int y = (int)infos[i].bbox.ymin;
					int w = (int)(infos[i].bbox.xmax - infos[i].bbox.xmin + 1);
					int h = (int)(infos[i].bbox.ymax - infos[i].bbox.ymin + 1);
					float score = infos[i].bbox.score;
					x = x - (h - w) / 2;
					h = h * 0.85f;
					w = h;
					y = y + 0.2 * h;
					array<int>^ landmark = gcnew array<int>(10);
					for (int j = 0; j < 5; j++)
					{
						landmark[2 * j] = (int)infos[i].landmark[2 * j];
						landmark[2 * j + 1] = (int)infos[i].landmark[2 * j + 1];
					}
					float lefteye_nose = Math::Abs(infos[i].landmark[0] - infos[i].landmark[4]);
					float righteye_nose = Math::Abs(infos[i].landmark[1] - infos[i].landmark[4]);
					float arc_yaw = Math::Atan(lefteye_nose / righteye_nose / 10);
					float radius_yaw = arc_yaw / 3.1415 * 180;
					x += w * radius_yaw * 0.01;
					output->Add(FaceInfo(i, radius_yaw, score, System::Drawing::Rectangle(x, y, w, h), landmark));
				}
				return output;
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
				for(int i = 0; i < n; i++)
				{
					output->Add(FaceInfo(i, 0, scores[i], System::Drawing::Rectangle(X[i], Y[i], S[i], S[i])));
				}
				delete[] buf;
				return output;
			}
			
		}
	}
}