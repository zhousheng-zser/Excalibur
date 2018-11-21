#ifndef _TENSOR_UTILS_HPP_
#define _TENSOR_UTILS_HPP_
#include <glog/logging.h>
namespace excalibur
{
	enum interpolationType { Nearest, Bilinear, Cubic };

	enum borderType { Border_Constant, Border_Replicate };

	enum flipType { Channel_Wise, Width_Wise, Height_Wise, Center_Wise };

	enum thresholdType { binary, binary_inv, small_trunc, big_trunc, small_to_zero, big_to_zero };

	enum morphType { Dilate, Erode };

	enum tensorType { NHWC, NCHW };

	///Rotation_Invariant(RI)
	enum lbpType { Native, RI, U2, RIU2, HF, LTP };

	template <typename Dtype>
	class point
	{
	public:
		Dtype x;
		Dtype y;

		point()
		{
			x = Dtype(0);
			y = Dtype(0);
		}

		point(Dtype x, Dtype y)
		{
			this->x = x;
			this->y = y;
		}

		point& operator=(const point& r)
		{
			if (this == &r)
			{
				return *this;
			}
			x = r.x;
			y = r.y;
			return *this;
		}

		point(const point& r)
		{
			x = r.x;
			y = r.y;
		}

		float distance(const point& r)
		{
			return sqrt((x - r.x) * (x - r.x) * 1.0f+ (y - r.y) * (y - r.y) * 1.0f);
		}
	};

	template <typename Dtype>
	class rectangle
	{
	public:
		Dtype x;
		Dtype y;
		Dtype h;
		Dtype w;

		rectangle()
		{
			this->x = Dtype(0);
			this->y = Dtype(0);
			this->h = Dtype(0);
			this->w = Dtype(0);
		}

		rectangle(Dtype x, Dtype y, Dtype h, Dtype w)
		{
			CHECK_GE(h, 0);
			CHECK_GE(w, 0);
			this->x = x;
			this->y = y;
			this->h = h;
			this->w = w;
		}

		rectangle(point<Dtype> top_left, point<Dtype> bottom_right)
		{
			this->x = top_left.x;
			this->y = top_left.y;
			this->h = bottom_right.y - top_left.y;
			this->w = bottom_right.x - top_left.x;
			CHECK_GE(h, 0);
			CHECK_GE(w, 0);
		}

		rectangle& operator=(const rectangle& r)
		{
			if (this == &r)
			{
				return *this;
			}
			x = r.x;
			y = r.y;
			h = r.h;
			w = r.w;
			return *this;
		}

		rectangle(const rectangle& r)
		{
			x = r.x;
			y = r.y;
			h = r.h;
			w = r.w;
		}

		Dtype IoU(const rectangle r)
		{
			return Dtype(0);// Todo
		}

		point<Dtype> center()
		{
			return point<Dtype>(Dtype(x + w * 0.5f), Dtype(y + h * 0.5f));
		}
	};

	
	class color
	{
	public:
		unsigned char r;
		unsigned char g;
		unsigned char b;

		color()
		{
			r = (unsigned char)0;
			g = (unsigned char)0;
			b = (unsigned char)0;
		}

		template <typename Dtype>
		color(Dtype r, Dtype g, Dtype b)
		{
			this->r = (unsigned char)r;
			this->g = (unsigned char)g;
			this->b = (unsigned char)b;
		}


	};
}
#endif // !_TENSOR_UTILS_HPP_
