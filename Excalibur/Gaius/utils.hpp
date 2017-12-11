#pragma once
#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include "ImageTensor.hpp"

namespace excalibur
{
	class utils
	{
	public:
		static void NonMaximumSuppression(std::vector<FaceInfo>* bboxes,
			std::vector<FaceInfo>* bboxes_nms, float iou_thresh = 0.8f);
	};
}


#endif // _UTILS_HPP_