#pragma once
#ifndef _ATHENE_HPP_
#define _ATHENE_HPP_

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"

struct BlobData;
namespace glasssix 
{
	namespace athene
	{
		class Athene
		{
		public:
			Athene() {};

			Athene(const char *deploy, const char *caffemodel, int base_height = 200, int base_width = 200,  int device = 0);

			void Forward(cv::Mat &image);

			~Athene();

		private:
			int base_height_;
			int base_width_;
			const char *deploy_;
			const char *caffemodel_;
			int device_;
			BlobData* nms_out_;
			BlobData* input_;
			cv::Size input_size_;
			cv::Size baseSize_;
		};
	}
}
#endif//!USE_OPENCV
#endif///!_ATHENE_HPP_