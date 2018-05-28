#ifndef _TENSOR_CV_UTILS_HPP_
#define _TENSOR_CV_UTILS_HPP_

namespace glasssix
{
	namespace gilgamesh
	{
		public enum class ImageEncodingType : int { Native, Bmp, Png, Jpeg };

		public enum class InterpolationType : int { Nearest, Bilinear, Cubic };

		public enum class BorderType : int { Border_Constant, Border_Replicate };

		public enum class FlipType : int { C_Wise, W_Wise, H_Wise };
	}
}
#endif // !_TENSOR_CV_UTILS_HPP_

