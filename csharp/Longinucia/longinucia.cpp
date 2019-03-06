#include "longinucia.hpp"

using namespace System::Drawing;
using namespace System::Drawing::Imaging;

namespace glasssix
{
	namespace longinus
	{
		Longinucia::Longinucia()
		{
			long_wrap = new LonginusDetector();
			image_data = nullptr;
			this->width = 0;
			this->height = 0;
		}

		Longinucia::Longinucia(int width, int height)
		{
			long_wrap = new LonginusDetector();
			image_data = new unsigned char[width * height];
			this->width = width;
			this->height = height;
		}

		Longinucia::~Longinucia()
		{
			this->!Longinucia();
		}

		Longinucia::!Longinucia()
		{
			delete long_wrap;
			long_wrap = nullptr;
			if (image_data)
			{
				delete image_data;
				image_data = nullptr;
			}
		}

		unsigned char* Longinucia::Bitmap2Gray(System::Drawing::Bitmap^ bmp)
		{
			int width = bmp->Width;
			int height = bmp->Height;
			if (this->width * this->height > 0)
			{
				if (width != this->width || height != this->height)
				{
					Console::WriteLine("Error input image size.");
					return nullptr;
				}
			}
			unsigned char* dst_data = new unsigned char[width * height];
			memset(dst_data, 0, width * height);
			System::Drawing::Imaging::BitmapData^ bmpd;
			//Gray8
			if (bmp->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)
			{
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, width, height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, bmp->PixelFormat);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				int stride = (width * 1 + 3) & -4;
				for (int h = 0; h < height; h++)
				{
					int offset = h * width;
					int h_stride = h * stride;
					for (int w = 0; w < width; w++)
					{
						dst_data[offset + w] = (unsigned char)pBmp[h_stride + w];
					}
				}
				bmp->UnlockBits(bmpd);
			}
			//else, convert to RGB24
			else
			{
				unsigned char* dst_data_3c = new unsigned char[width * height * 3];
				bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height),
					System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
				unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
				int stride =  (width * 3 + 3) & -4;
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
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width;
					for (int w = 0; w < width; w++)
					{
						dst_data[sub_offset + w] = 
							static_cast<unsigned char>(dst_data_3c[offset * 0 + sub_offset + w] * 0.33f + dst_data_3c[offset * 1 + sub_offset + w] * 0.33f + dst_data_3c[offset * 2 + sub_offset + w] * 0.33f);
					} 
				}
				bmp->UnlockBits(bmpd);
				delete dst_data_3c;
			}
			if (image_data) // for data multiplexing
			{
				memcpy(image_data, dst_data, this->width * this->height);
			}
			return dst_data;
		}

		unsigned char* Longinucia::Bitmaps2RGB(System::Drawing::Bitmap^ bmp)
		{
			int channel = 3;
			int width = bmp->Width;
			int height = bmp->Height;
			if (this->width * this->height > 0)
			{
				if (width != this->width || height != this->height)
				{
					Console::WriteLine("Error input image size.");
					return nullptr;
				}
			}
			if (width <= 0 || height <= 0)
			{
				return nullptr;
			}
			int c_offset = height * width;
			unsigned char* dst_data = new unsigned char[channel * c_offset];
			System::Drawing::Imaging::BitmapData^ bmpd;
			bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, width, height),
				System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
			unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
			int stride = (width * 3 + 3) & -4;
			int offset = width * height;
			if (image_data) // for data multiplexing
			{
				memset(image_data, 0, width * height);
				for (int c = 0; c < 3; c++)
				{
					int c_offset = offset * c;
					for (int h = 0; h < height; h++)
					{
						int sub_offset = h * width;
						int h_stride = h * stride;
						for (int w = 0; w < width; w++)
						{
							dst_data[c_offset + sub_offset + w]
								= (unsigned char)pBmp[h_stride + w * 3 + c];
							image_data[sub_offset + w] += static_cast<unsigned char>(dst_data[c_offset + sub_offset + w] * 0.33f);
						}
					}
				}
			}
			else
			{
				for (int c = 0; c < 3; c++)
				{
					int c_offset = offset * c;
					for (int h = 0; h < height; h++)
					{
						int sub_offset = h * width;
						int h_stride = h * stride;
						for (int w = 0; w < width; w++)
						{
							dst_data[c_offset + sub_offset + w]
								= (unsigned char)pBmp[h_stride + w * 3 + c];
						}
					}
				}
			}
			bmp->UnlockBits(bmpd);
			return dst_data;
		}

		array<System::Drawing::Bitmap^>^ Longinucia::Uchar2Bitmaps(unsigned char* data, int num, int channel, int height, int width)
		{
			array<System::Drawing::Bitmap^>^ outputs = gcnew array<Bitmap^>(num);
			for (size_t n = 0; n < num; n++)
			{
				outputs[n] = gcnew System::Drawing::Bitmap(width, height, PixelFormat::Format24bppRgb);
				BitmapData^ bmpdata = outputs[n]->LockBits(
					System::Drawing::Rectangle(System::Drawing::Point::Empty, Size(width, height)),
					ImageLockMode::WriteOnly,
					PixelFormat::Format24bppRgb);
				unsigned char* dst_data = (unsigned char*)bmpdata->Scan0.ToPointer();
				int offset = 3 * width;
				int stride = width * channel;
				int src_c_stride = width * height;
				int src_n_stride = channel * src_c_stride;
				int aligned_stride = (width * channel + 3) & -4;
				int dst_aligned_offset = 0;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * offset;
					int aligned_stride_h = aligned_stride * h;
					for (int w = 0; w < width; w++)
					{
						for (int c = 0; c < channel; c++)
						{
							dst_data[aligned_stride_h + w * channel + c] =
								data[n * src_n_stride + c * src_c_stride + h * width + w];
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
				outputs[n]->UnlockBits(bmpdata);
			}
			return outputs;
		}

		List<FaceInfo>^ Longinucia::Face_Detect(System::Drawing::Bitmap^ bmp, int min_size, float scale,
			int minNeighbors, bool useMultiThreads, bool doEarlyReject, bool doLandmark)
		{
			List<FaceInfo>^ output = gcnew List<FaceInfo>();
			auto data = Bitmap2Gray(bmp);
			if (!data)
			{
				return output;
			}
			if (doLandmark)
			{
				auto res = long_wrap->detect(data, bmp->Width, bmp->Height, bmp->Width,
					min_size, scale, minNeighbors, 0, useMultiThreads, doEarlyReject);
				for (size_t i = 0; i < res.size(); i++)
				{
					array<int>^ ldmk = gcnew array<int>(10);
					for (size_t j = 0; j < 5; j++)
					{
						ldmk[2 * j + 0] = (int)(res[i].pts[j].x);
						ldmk[2 * j + 1] = (int)(res[i].pts[j].y);
					}
					output->Add(FaceInfo(System::Drawing::Rectangle(res[i].x, res[i].y, res[i].width, res[i].height),
						res[i].yaw, res[i].pitch, res[i].roll, res[i].confidence, ldmk));
				}
			}
			else
			{
				auto res = long_wrap->detect(data, bmp->Width, bmp->Height, bmp->Width, 
					min_size, scale, minNeighbors, useMultiThreads, doEarlyReject);
				for (size_t i = 0; i < res.size(); i++)
				{
					output->Add(FaceInfo(System::Drawing::Rectangle(res[i].x, res[i].y, res[i].width, res[i].height)));
				}
			}
			delete data;
			return output;
		}

		List<FaceInfo>^ Longinucia::Face_DetectEx(System::Drawing::Bitmap^ bmp, int min_size, float scale, array<float>^ thresholds, int stage)
		{
			List<FaceInfo>^ output = gcnew List<FaceInfo>();
			if ((stage != thresholds->Length) || (stage > 3))
			{
				return output;
			}
			auto data = Bitmaps2RGB(bmp);
			if (!data)
			{
				return output;
			}
			float th[3];
			for (auto i = 0; i < stage; i++)
			{
				th[i] = thresholds[i];
			}
			auto res = long_wrap->detectEx(data, 3, bmp->Height, bmp->Width, min_size, th, 1.0f / scale, stage);
			delete data;
			for (size_t i = 0; i < res.size(); i++)
			{
				array<int>^ ldmk = gcnew array<int>(10);
				for (size_t j = 0; j < 5; j++)
				{
					ldmk[2 * j + 0] = (int)(res[i].pts[j].x);
					ldmk[2 * j + 1] = (int)(res[i].pts[j].y);
				}
				output->Add(FaceInfo(System::Drawing::Rectangle(res[i].x, res[i].y, res[i].width, res[i].height),
					res[i].yaw, res[i].pitch, res[i].roll, res[i].confidence, ldmk));
			}
			return output;
		}

		void Longinucia::set(DetectorType type, int device)
		{
			switch (type)
			{
			case glasssix::longinus::DetectorType::FRONTALVIEW:
				long_wrap->set(glasssix::longinus::DetectionType::FRONTALVIEW, device);
				break;
			case glasssix::longinus::DetectorType::FRONTALVIEW_REINFORCE:
				long_wrap->set(glasssix::longinus::DetectionType::FRONTALVIEW_REINFORCE, device);
				break;
			case glasssix::longinus::DetectorType::MULTIVIEW:
				long_wrap->set(glasssix::longinus::DetectionType::MULTIVIEW, device);
				break;
			case glasssix::longinus::DetectorType::MULTIVIEW_REINFORCE:
				long_wrap->set(glasssix::longinus::DetectionType::MULTIVIEW_REINFORCE, device);
				break;
			default:
				long_wrap->set(glasssix::longinus::DetectionType::FRONTALVIEW, device);
				break;
			}
		}

		void Longinucia::Match_Faces(List<FaceInfo>^% infos, int frame_extract_frequency)
		{
			std::vector<FaceRect> rects;
			for (size_t i = 0; i < infos->Count; i++)
			{
				rects.push_back(FaceRect(infos[i].rect.X, infos[i].rect.Y, infos[i].rect.Width, infos[i].rect.Height, 1, 1.0));
			}
			auto res = long_wrap->match(rects, frame_extract_frequency);
			for (size_t i = 0; i < res.size(); i++)
			{
				infos[i] = FaceInfo(infos[i].rect, infos[i].yaw, infos[i].pitch, infos[i].roll, infos[i].score,
					infos[i].landmarks, gcnew String(res[i].id.c_str()), res[i].is_new);
			}
		}

		array<System::Drawing::Bitmap^>^ Longinucia::AlignFace(List<FaceInfo>^ infos)
		{
			if (this->width * this->height <= 0)
			{
				Console::WriteLine("No data multiplexing, use another 'AlignFace' function.");
				return gcnew array<System::Drawing::Bitmap^>(0);
			}
			std::vector<std::vector<int>> bboxs;
			std::vector<std::vector<int>> landmarks;
			for (size_t i = 0; i < infos->Count; i++)
			{
				std::vector<int> bbox = { infos[i].rect.X, infos[i].rect.Y,infos[i].rect.Height,infos[i].rect.Width };
				std::vector<int> landmark;
				for (size_t j = 0; j < 10; j++)
				{
					int temp = infos[i].landmarks[j];
					landmark.push_back(temp);
				}
				bboxs.push_back(bbox);
				landmarks.push_back(landmark);
			}
			auto res = long_wrap->alignFace(image_data, 1, 1, height, width, bboxs, landmarks);
			return Uchar2Bitmaps(res.data(), infos->Count, 3, 128, 128);
		}

		array<System::Drawing::Bitmap^>^ Longinucia::AlignFace(System::Drawing::Bitmap^ bmp, List<FaceInfo>^ infos)
		{
			std::vector<std::vector<int>> bboxs;
			std::vector<std::vector<int>> landmarks;
			for (size_t i = 0; i < infos->Count; i++)
			{
				std::vector<int> bbox = { infos[i].rect.X, infos[i].rect.Y,infos[i].rect.Height,infos[i].rect.Width };
				std::vector<int> landmark;
				for (size_t j = 0; j < 10; j++)
				{
					int temp = infos[i].landmarks[j];
					landmark.push_back(temp);
				}
				bboxs.push_back(bbox);
				landmarks.push_back(landmark);
			}
			auto data = Bitmap2Gray(bmp);
			auto res = long_wrap->alignFace(data, 1, 1, bmp->Height, bmp->Width, bboxs, landmarks);
			delete data;
			return Uchar2Bitmaps(res.data(), infos->Count, 3, 128, 128);
		}
	}
}