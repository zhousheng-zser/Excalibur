#include "athenel.hpp"

#include <msclr\marshal_cppstd.h>

namespace glasssix
{
	namespace athene
	{
		Athenel::Athenel(System::String^ stream, System::String^ deploy, System::String^ caffemodel, int base_height, int base_width, int device)
		{
			pose_profiler = new Athene(msclr::interop::marshal_as<std::string>(stream).c_str(), msclr::interop::marshal_as<std::string>(deploy).c_str(), msclr::interop::marshal_as<std::string>(caffemodel).c_str(), base_height, base_width, device);
		}

		Athenel::~Athenel()
		{
			this->!Athenel();
		}

		Athenel::!Athenel()
		{
			delete pose_profiler;
			pose_profiler = nullptr;
		}

		cv::Mat Athenel::Bitmap2Mat(Bitmap^ bmp)
		{
			int channel = 3;
			int width = bmp->Width;
			int height = bmp->Height;
			cv::Mat result = cv::Mat(height, width, CV_8UC3);

			System::Drawing::Imaging::BitmapData^ bmpd;
			bmpd = bmp->LockBits(System::Drawing::Rectangle(0, 0, width, height),
				System::Drawing::Imaging::ImageLockMode::ReadOnly, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
			unsigned char* pBmp = (unsigned char*)bmpd->Scan0.ToPointer();
			memcpy(result.data, pBmp, height * width * channel * sizeof(unsigned char));
			bmp->UnlockBits(bmpd);

			return result;
		}

		Bitmap^ Athenel::Mat2Bitmap(cv::Mat image)
		{
			int channel = image.channels();
			int width = image.cols;
			int height = image.rows;

			Bitmap^ outputs = gcnew System::Drawing::Bitmap(width, height, PixelFormat::Format24bppRgb);
			BitmapData^ bmpdata = outputs->LockBits(
				System::Drawing::Rectangle(System::Drawing::Point::Empty, Size(width, height)),
				ImageLockMode::WriteOnly,
				PixelFormat::Format24bppRgb);
			unsigned char* dst_data = (unsigned char*)bmpdata->Scan0.ToPointer();
			memcpy(dst_data, image.data, height * width * channel * sizeof(unsigned char));
			outputs->UnlockBits(bmpdata);

			return outputs;
		}

		Bitmap^ Athenel::Forward(Bitmap^ imgData)
		{
			cv::Mat image = Bitmap2Mat(imgData);
			pose_profiler->Forward(image);
			return Mat2Bitmap(image);
		}

		void Athenel::Forward()
		{
			pose_profiler->Forward();
		}
	}
}