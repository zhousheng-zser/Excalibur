#ifndef _TENSOR_CV_UTILS_HPP_
#define _TENSOR_CV_UTILS_HPP_
#include "../Excalibur/tensor_utils.hpp"

namespace glasssix
{
	namespace gilgamesh
	{
		public enum class ImageEncodingType : int { Native, Bmp, Png, Jpeg };

		public enum class InterpolationType : int { Nearest, Bilinear, Cubic };

		public enum class BorderType : int { Border_Constant, Border_Replicate };

		public enum class FlipType : int { C_Wise, W_Wise, H_Wise };

		public ref class rectangle
		{
		public:
			excalibur::rectangle<int> *rect;

			rectangle()
			{
				rect = new excalibur::rectangle<int>();
			}

			!rectangle()
			{
				delete rect;
				rect = nullptr;
			}

			~rectangle()
			{
				this->!rectangle();
			}

			rectangle(const rectangle %r)
			{
				rect = r.rect;
			}

			rectangle(int x, int y, int h, int w)
			{
				rect = new excalibur::rectangle<int>(x, y, h, w);
			}
		};

		public ref class color
		{
		public:
			excalibur::color *c;

			color()
			{
				c = new excalibur::color();
			}

			!color()
			{
				delete c;
				c = nullptr;
			}

			~color()
			{
				this->!color();
			}

			color(const color %c_)
			{
				c = c_.c;
			}

			color(unsigned char b, unsigned char g, unsigned char r)
			{
				// To keep compatibility with OpenCV(RGB2BGR)
				c = new excalibur::color(r, g, b);
			}
		};
	}
}
#endif // !_TENSOR_CV_UTILS_HPP_

