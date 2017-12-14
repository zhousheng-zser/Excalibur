#pragma once
#ifndef _DETECTOR_HPP_
#define _DETECTOR_HPP_

#include "ImagePyramid.hpp"

namespace excalibur
{
	class Detector {
	public:
		Detector() {}
		virtual ~Detector() {}

		virtual bool LoadModel(const std::string & model_path) = 0;
		virtual std::vector<FaceInfo> Detect(ImagePyramid* img_pyramid) = 0;

		virtual void SetWindowSize(int size) {}
		virtual void SetSlideWindowStep(int step_x, int step_y) {}

		//DISABLE_COPY_AND_ASSIGN(Detector);
	};
}

#endif // _DETECTOR_HPP_