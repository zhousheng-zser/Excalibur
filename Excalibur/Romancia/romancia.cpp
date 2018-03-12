#include "romancia.hpp"

namespace glasssix
{
	namespace aroundight
	{
		namespace romancia
		{

			FastCNNFace::FastCNNFace(int device)
			{
				mw_ = new mtcnn_warpper(device);
			}

			FastCNNFace::~FastCNNFace()
			{
				this->!FastCNNFace();
			}

			FastCNNFace::!FastCNNFace()
			{
				delete mw_;
				mw_ = NULL;
			}

			unsigned char* Bitmap2RGB(System::Drawing::Bitmap^ bmp)
			{
				int stride;
				unsigned char* res;
				System::Drawing::Imaging::BitmapData^ bmpd;
				if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed) {
					stride = bmpd->Stride;
					bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
					return (unsigned char*)bmpd->Scan0.ToPointer();
				}
				else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
				{
					bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
					stride = bmp->Width + (4 - bmp->Width % 4) % 4;
					res = new unsigned char[stride * bmp->Height * 3];
					unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer(),
						*b, *g, *r;
					for (int offset = 0, y = 0; y < bmp->Height; ++y, offset += bmpd->Stride) {
						b = pBmp + offset + 0, g = pBmp + offset + 1, r = pBmp + offset + 2;
						for (int x = 0; x < bmpd->Width; ++x, b += 3, g += 3, r += 3) {
							//res[y * stride + x] = (unsigned char)((float)*r * 0.3f + (float)*g * 0.59f + (float)*b * 0.11f);
							//res[y * stride + x] = (unsigned char)(((s1mul0_3((int)*r)) + (s1mul0_59((int)*g)) + (s1mul0_11((int)*b))) >> 1);
							res[(y * stride + x) * 3] = *b;
							res[(y * stride + x) * 3 + 1] = *g;
							res[(y * stride + x) * 3 + 2] = *r;
						}
					}
					bmp->UnlockBits(bmpd);

				}
				else
				{
					res = nullptr;
				}
				return res;
			}

			List<FaceInfo>^ FastCNNFace::Facedetect_Mtcnn(System::Drawing::Bitmap^ Oribmp, int min_size)
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
					output->Add(FaceInfo(i, 0, score, System::Drawing::Rectangle(x, y, w, h)));
				}
				return output;
			}
		}
	}
}