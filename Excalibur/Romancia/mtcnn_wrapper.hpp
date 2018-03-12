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

	private:
		MTCNN* mt_;
		int device_;
		//
		float factor = 0.709f;
		float threshold[3] = { 0.7f, 0.6f, 0.6f };
	};

	

}

#endif