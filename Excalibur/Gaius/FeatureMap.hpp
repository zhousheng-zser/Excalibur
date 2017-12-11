#pragma once
#ifndef _FEATUREMAP_HPP_
#define _FEATUREMAP_HPP_

#include "ImageTensor.hpp"

namespace excalibur
{
	class FeatureMap
	{
	public:
		FeatureMap()
			: width_(0), height_(0) {
			roi_.x = 0;
			roi_.y = 0;
			roi_.width = 0;
			roi_.height = 0;
		}

		virtual ~FeatureMap() {}

		virtual void ComputeCPU(const uint8_t* input, int32_t width, int32_t height) = 0;

		virtual void ComputeGPU(const uint8_t* input, int32_t width, int32_t height) = 0;

		inline virtual void SetROI(const Rect & roi) {roi_ = roi;}

	protected:
		int32_t width_;
		int32_t height_;
		Rect roi_;
	};
}


#endif // _FEATUREMAP_HPP_