#pragma once
#include <msclr\marshal_cppstd.h>
#include "../Excalibur/tensoroperation.hpp"
#include "../Excalibur/tensor_utils.hpp"

using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
//using namespace System::Drawing;
//using namespace System::Drawing::Imaging;

namespace glasssix
{
	namespace gilgamesh
	{
			public ref class Tensor
			{
				int num;
				int channel;
				int width;
				int height;
				int device;
				
				excalibur::tensor<float>* data;

			public:
				Tensor(){};

				property int Num
				{
					int get()
					{
						return num;
					}
				}
				property int Channel
				{
					int get()
					{
						return channel;
					}
				}
				property int Width
				{
					int get()
					{
						return width;
					}
				}
				property int Height
				{
					int get()
					{
						return height;
					}
				}
				property int Device
				{
					int get()
					{
						return device;
					}
				}
			};
	}
}