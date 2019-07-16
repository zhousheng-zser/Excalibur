#pragma once
#ifndef _LONGINUCIA_HPP_
#define _LONGINUCIA_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <msclr/marshal.h>
#include "../../include/Longinus/LonginusDetector.hpp"

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Globalization;
using namespace System::IO;
using namespace System::Runtime::InteropServices;
using namespace msclr::interop;

namespace glasssix
{
	namespace longinus
	{
		public enum class DetectorType
		{
			FRONTALVIEW,
			FRONTALVIEW_REINFORCE,
			MULTIVIEW,
			MULTIVIEW_REINFORCE
		};

		public value struct FaceInfo
		{
			float yaw;
			float pitch;
			float roll;
			float score;
			System::Drawing::Rectangle rect;
			array<int>^ landmarks;
			String^ guid;
			bool new_appearance;

			FaceInfo(System::Drawing::Rectangle rect) :rect(rect) 
			{
				yaw = 0.0f;
				pitch = 0.0f;
				roll = 0.0f;
				score = 0.0f;
				new_appearance = false;
			};

			FaceInfo(System::Drawing::Rectangle rect, float yaw, float pitch, float roll, float score) 
				:rect(rect), yaw(yaw), pitch(pitch), roll(roll), score(score) 
			{
				new_appearance = false;
			}

			FaceInfo(System::Drawing::Rectangle rect, float yaw, float pitch, float roll, float score, array<int>^ landmarks)
				:rect(rect), yaw(yaw), pitch(pitch), roll(roll), score(score), landmarks(landmarks)
			{
				new_appearance = false;
			}

			FaceInfo(System::Drawing::Rectangle rect, float yaw, float pitch, float roll, float score, array<int>^ landmarks, String^ guid, bool new_appearance)
				:rect(rect), yaw(yaw), pitch(pitch), roll(roll), score(score), landmarks(landmarks), guid(guid), new_appearance(new_appearance)
			{
				
			}
		};

		public ref class Longinucia
		{
			LonginusDetector* long_wrap;
			unsigned char* Bitmap2Gray(System::Drawing::Bitmap^ bmp);
			unsigned char* Bitmaps2RGB(System::Drawing::Bitmap^ bmp);
			array<System::Drawing::Bitmap^>^ Uchar2Bitmaps(unsigned char* data, int num, int channel, int height, int width);
			unsigned char* image_data;
			int width;
			int height;
		public:
			Longinucia();
			// For survillence video.
			Longinucia(int width, int height);
			~Longinucia();
			!Longinucia();

			void set(DetectorType type, int device);

			List<FaceInfo>^ Face_Detect(System::Drawing::Bitmap^ bmp, int min_size, float scale, 
				int minNeighbors, bool useMultiThreads, bool doEarlyReject, bool doLandmark);

#ifndef TRIAL
			List<FaceInfo>^ Face_DetectEx(System::Drawing::Bitmap^ bmp, int min_size, float scale, array<float>^ thresholds, int stage);

			System::Drawing::Bitmap^ AlignFace(System::Drawing::Bitmap^ extend_face_bmp);
#endif // !TRIAL

			void Match_Faces(List<FaceInfo>^% infos, int frame_extract_frequency);

			array<System::Drawing::Bitmap^>^ AlignFace(List<FaceInfo>^ infos);

			array<System::Drawing::Bitmap^>^ AlignFace(System::Drawing::Bitmap^ bmp, List<FaceInfo>^ infos);
		};
	}
}
#endif // !_LONGINUCIA_HPP_
