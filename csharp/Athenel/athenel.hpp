#pragma once
#ifndef _POSE_PROFILERIA_HPP_
#define _POSE_PROFILERIA_HPP_
#include <msclr\marshal_cppstd.h>
#include <msclr\marshal.h>
#include "../../include/Athene/athene.hpp"

using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;
using namespace msclr::interop;

namespace glasssix
{
	namespace athene
	{
		public ref class Athenel
		{
		public:
			Athenel(System::String^ deploy, System::String^ caffemodel, int base_height, int base_width, int device);
			~Athenel();
			!Athenel();

			Bitmap^ Forward(Bitmap^ imgData);

		private:
			Athene* pose_profiler;

			cv::Mat Bitmap2Mat(Bitmap^ bmp);

			Bitmap^ Mat2Bitmap(cv::Mat image);
		};
	}
}
#endif