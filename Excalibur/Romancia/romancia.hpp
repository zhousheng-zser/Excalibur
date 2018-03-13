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

#include "mtcnn_wrapper.hpp"

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
				mtcnn_warpper* mw_;

				//float scale;//1.2f
				float threshold_;//3
				//int min_object_width;//24
				bool dolandmark_;//true

				List<FaceInfo>^ GetResultsList(int * pResults);
				unsigned char* Bitmap2Gray(System::Drawing::Bitmap^ bmp, int& stride);
				unsigned char* Bitmap2RGB(System::Drawing::Bitmap^ bmp);
			public:
				FastFace(int device);
				void SetLDMKDetector(bool dolandmark)
				{
					dolandmark_ = dolandmark;
				}
				void SetThreshold(float threshold)
				{
					threshold_ = threshold;
				}
				~FastFace();
				!FastFace();
				List<FaceInfo>^ Facedetect_Frontal(System::Drawing::Bitmap^ Oribmp, int min_size, float scale);
				List<FaceInfo>^ Facedetect_Multiview(System::Drawing::Bitmap^ Oribmp, int min_size, float scale);
				List<FaceInfo>^ Facedetect_Multiview_Reinforce(System::Drawing::Bitmap^ Oribmp, int min_size, float scale);
				List<FaceInfo>^ Facedetect_Frontal_Surveillance(System::Drawing::Bitmap^ Oribmp, int min_size, float scale);
				List<FaceInfo>^ Facedetect_Multiview_CNN(System::Drawing::Bitmap^ Oribmp, int min_size, float scale);
			};
		}
	}
}

#endif