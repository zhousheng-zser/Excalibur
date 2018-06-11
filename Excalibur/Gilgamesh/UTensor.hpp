#ifndef _IMAGE_CLI_HPP_
#define _IMAGE_CLI_HPP_

#include <msclr\marshal_cppstd.h>
#include "../Excalibur/utensor.hpp"
#include "TensorCVUtils.hpp"

using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;
// #pragma make_public directive is currently supported for native non-template types only
#pragma make_public(excalibur::utensor) 

namespace glasssix
{
	namespace gilgamesh
	{

		public ref class UTensor
		{
			int num;
			int channel;
			int width;
			int height;
			int device;

		public:
			excalibur::utensor* data;

			UTensor();

			!UTensor();

			~UTensor();

			// (deep) copy construct function
			UTensor(const UTensor %t);

			UTensor(int num, int channel, int width, int height, int device);

			UTensor(int channel, int width, int height, int device);

			UTensor(int width, int height, int device);

			UTensor(int size, int device);

			UTensor(Bitmap^ bmp, int device);

			UTensor(String^ path, int device);

			Bitmap^ ToBitmap();

			void Save(String^ path, ImageEncodingType type);

			property int Num
			{
				int get()
				{
					return num;
				}
			private:
				void set(int num)
				{
					this->num = num;
				}
			}

			property int Channel
			{
				int get()
				{
					return channel;
				}
			private:
				void set(int channel)
				{
					this->channel = channel;
				}
			}

			property int Width
			{
				int get()
				{
					return width;
				}
			private:
				void set(int width)
				{
					this->width = width;
				}
			}

			property int Height
			{
				int get()
				{
					return height;
				}
			private:
				void set(int height)
				{
					this->height = height;
				}
			}

			property int Device
			{
				int get()
				{
					return device;
				}
			private:
				void set(int device)
				{
					this->device = device;
				}
			}

		};
	}
}

#endif // !_IMAGE_CLI_HPP_