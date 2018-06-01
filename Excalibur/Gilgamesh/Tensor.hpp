#ifndef _TENSOR_CLI_HPP_
#define _TENSOR_CLI_HPP_

#include <msclr\marshal_cppstd.h>
#include "../Excalibur/tensor.hpp"
#include "TensorCVUtils.hpp"

using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;

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

		public:
			excalibur::tensor<float>* data;

			Tensor();

			!Tensor();

			~Tensor();

			// (deep) copy construct function
			Tensor(const Tensor %t);

			Tensor(int num, int channel, int width, int height, int device);

			Tensor(int channel, int width, int height, int device);

			Tensor(int width, int height, int device);

			Tensor(int size, int device);

			Tensor(Bitmap^ bmp, int device);

			Tensor(String^ path, int device);

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

#endif // !_TENSOR_CLI_HPP_