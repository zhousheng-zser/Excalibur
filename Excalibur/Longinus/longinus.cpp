#include <msclr\marshal_cppstd.h>
#include "alignment.hpp"

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
			public value struct AlignFaceInfo
			{
				int yaw;
				int pitch;
				int roll;
				System::Drawing::Bitmap^ align_face;
				AlignFaceInfo(Bitmap^ AF, int Y, int P, int R) :align_face(AF), yaw(Y), pitch(P), roll(R) {};
			};

			public ref class Banshee
			{
			private:
				int device_;
				alignment* aligner_;
			public:
				Banshee(int device)
				{
					device_ = device;
					aligner_ = new alignment(device_);
				}
				!Banshee()
				{
					delete aligner_;
					aligner_ = nullptr;
				}

				~Banshee()
				{
					this->!Banshee();
				}

				System::Drawing::Bitmap^ align(System::Drawing::Bitmap^ detected_face)
				{
					int stride;
					unsigned char * buf = Bitmap2RGB(detected_face);
					cv::Mat detected_face_mat = cv::Mat(detected_face->Height, detected_face->Width, CV_8UC3, buf);
					cv::Mat aligned_face;
					aligner_->alignment_face(detected_face_mat, aligned_face);
					delete buf;
					return MatToBitmap(aligned_face);
				}

				List<AlignFaceInfo>^ align(array<Bitmap^>^ detected_faces)
				{
					List<AlignFaceInfo>^ output = gcnew List<AlignFaceInfo>();
					std::vector<cv::Mat> detected_face_mats(detected_faces->Length);
					std::vector<cv::Mat> aligned_faces(detected_faces->Length);
					std::vector<unsigned char *> bufs(detected_faces->Length);
					for (size_t i = 0; i < detected_faces->Length; i++)
					{
						int stride;
						bufs[i] = Bitmap2RGB(detected_faces[i]);
						detected_face_mats[i] = cv::Mat(detected_faces[i]->Height, detected_faces[i]->Width, CV_8UC3, bufs[i]);
					}
					aligner_->alignment_face(detected_face_mats, aligned_faces);
					auto yaw = aligner_->get_yaw_angle();
					auto pitch = aligner_->get_pitch_angle();
					auto roll = aligner_->get_roll_angle();
					for (size_t i = 0; i < detected_faces->Length; i++)
					{
						delete bufs[i];
						output->Add(AlignFaceInfo(MatToBitmap(aligned_faces[i]), yaw[i], pitch[i], roll[i]));
					}
					return output;
				}

			private:
				unsigned char* Bitmap2RGB(System::Drawing::Bitmap^ bmp)
				{
					int stride;
					unsigned char* res;
					System::Drawing::Imaging::BitmapData^ bmpd;
					if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)
					{
						stride = bmpd->Stride;
						bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), 
							System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
						return (unsigned char*)bmpd->Scan0.ToPointer();
					}
					else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
					{
						bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height), 
							System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
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
					else if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)
					{
						bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
							System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
						stride = bmp->Width;
						res = new unsigned char[stride * bmp->Height * 3];
						unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer(),
							*b, *g, *r;
						for (int offset = 0, y = 0; y < bmp->Height; ++y, offset += bmpd->Stride)
						{
							b = pBmp + offset + 0, g = pBmp + offset + 1, r = pBmp + offset + 2;
							for (int x = 0; x < bmpd->Width; ++x, b += 4, g += 4, r += 4)
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

				System::Drawing::Bitmap^ MatToBitmap(cv::Mat srcImg) {
					int stride = srcImg.size().width * srcImg.channels();//calc the srtide
					int hDataCount = srcImg.size().height;

					System::Drawing::Bitmap^ retImg;

					System::IntPtr ptr(srcImg.data);

					//create a pointer with Stride
					if (stride % 4 != 0) 
					{
						//is not stride a multiple of 4?
						//make it a multiple of 4 by fiiling an offset to the end of each row

						//to hold processed data
						uchar *dataPro = new uchar[((srcImg.size().width * srcImg.channels() + 3) & -4) * hDataCount];

						uchar *data = srcImg.ptr();

						//current position on the data array
						int curPosition = 0;
						//current offset
						int curOffset = 0;

						int offsetCounter = 0;

						//itterate through all the bytes on the structure
						for (int r = 0; r < hDataCount; r++) {
							//fill the data
							for (int c = 0; c < stride; c++) {
								curPosition = (r * stride) + c;

								dataPro[curPosition + curOffset] = data[curPosition];
							}

							//reset offset counter
							offsetCounter = stride;

							//fill the offset
							do {
								curOffset += 1;
								dataPro[curPosition + curOffset] = 0;

								offsetCounter += 1;
							} while (offsetCounter % 4 != 0);
						}

						ptr = (System::IntPtr)dataPro;
						//set the data pointer to new/modified data array

						//calc the stride to nearest number which is a multiply of 4
						stride = (srcImg.size().width * srcImg.channels() + 3) & -4;

						retImg = gcnew System::Drawing::Bitmap(srcImg.size().width, srcImg.size().height,
							stride,
							System::Drawing::Imaging::PixelFormat::Format24bppRgb,
							ptr);
					}
					else 
					{

						//no need to add a padding or recalculate the stride
						retImg = gcnew System::Drawing::Bitmap(srcImg.size().width, srcImg.size().height,
							stride,
							System::Drawing::Imaging::PixelFormat::Format24bppRgb,
							ptr);
					}

					array<unsigned char>^ imageData;
					System::Drawing::Bitmap^ output;

					// Create the byte array.
					{
						System::IO::MemoryStream^ ms = gcnew System::IO::MemoryStream();
						retImg->Save(ms, System::Drawing::Imaging::ImageFormat::Png);
						imageData = ms->ToArray();
						delete ms;
					}

					// Convert back to bitmap
					{
						System::IO::MemoryStream^ ms = gcnew System::IO::MemoryStream(imageData);
						output = (System::Drawing::Bitmap^)System::Drawing::Bitmap::FromStream(ms);
					}

					return output;
				}
			};
		}
	}
}