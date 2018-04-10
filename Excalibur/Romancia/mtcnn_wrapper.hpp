#pragma once
#ifndef _MTCNN_WRAPPER_HPP_
#define _MTCNN_WRAPPER_HPP_

#include "../fastfacedetection/mtcnn.hpp"

namespace glasssix
{
	class mtcnn_warpper
	{
	public:
		mtcnn_warpper(int device);
		~mtcnn_warpper();

		std::vector<FaceInfoX> facedetect_mtcnn(unsigned char* image_data, int width, int height, int min_size);

		std::vector<FaceInfoX> facedetect_mtcnn(const cv::Mat &image, int min_size);
	private:
		MTCNN* mt_;
		int device_;
		//
		float factor = 0.707f;
		float threshold[3] = { 0.70f, 0.60f, 0.60f };
	};
}

#endif