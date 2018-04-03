// FastRomancia.hpp

#pragma once

#ifndef _ROMANCIA_HPP_
#define _ROMANCIA_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <msclr/marshal.h>

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Globalization;
using namespace System::IO;
using namespace System::Runtime::InteropServices;
using namespace msclr::interop;

#include "npd_wrapper.hpp"

namespace glasssix
{
	namespace aroundight
	{
		namespace romancia
		{
			public value struct FaceInfo
			{
				int index;
				int angle;
				float score;
				System::Drawing::Rectangle rect;
				array<int>^ landmarks;
				FaceInfo(int Index) :index(Index) {}
				FaceInfo(int Index, int Angle, float Score, System::Drawing::Rectangle Rect) :index(Index), angle(Angle), score(Score), rect(Rect) {}
				FaceInfo(int Index, int Angle, float Score, System::Drawing::Rectangle Rect, array<int>^ Landmarks) :index(Index), angle(Angle), score(Score), rect(Rect), landmarks(Landmarks) {}
			};

			public ref class FastFace
			{
				npd_wrapper* nw_;
				float threshold_;//3
				unsigned char* Bitmap2Gray(System::Drawing::Bitmap^ bmp, int& stride);
			public:
				FastFace(int device);
				void SetThreshold(float threshold)
				{
					threshold_ = threshold;
				}
				~FastFace();
				!FastFace();
				List<FaceInfo>^ Facedetect_Frontal_Reinforce(System::Drawing::Bitmap^ Oribmp, int min_size, float scale);
			};
		}
	}
}
#endif