#pragma once
#ifndef _CASSIUNIA_HPP_
#define _CASSIUNIA_HPP_

#include <msclr\marshal_cppstd.h>
#include <msclr\marshal.h>
#include "../../include/Cassius/cassius.hpp"

using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;
using namespace msclr::interop;

namespace glasssix
{
	namespace cassius
	{
		public ref class Cassiunia
		{
		public:
			Cassiunia(int device);
			~Cassiunia();
			!Cassiunia();

			array<float>^ ExtractBitmapOutputs(array<Bitmap^>^ imgDatas);

			static float CosineDistanceProb(array<float>^ feature1, array<float>^ feature2)
			{
				float output = 0.0f;
				if (feature1->Length != feature2->Length)
				{
					output = -1.0f;
				}
				else
				{
					output = innerproduct(feature1, feature2)
						/ Math::Sqrt(innerproduct(feature1, feature1) * innerproduct(feature2, feature2));
				}
				return output;
			}

		private:
			Cassius* cassius_wrapper;
			int device_;

			static double innerproduct(array<float>^ feature1, array<float>^ feature2)
			{
				double output = 0;
				for (size_t i = 0; i < feature1->Length; i++)
				{
					output += feature1[i] * feature2[i];
				}
				return output;
			}

			unsigned char* Bitmaps2RGB(array<Bitmap^>^ bmps);
		};
	}
}
#endif // !_CASSIUNIA_HPP_
