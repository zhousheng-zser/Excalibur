#pragma once
#ifndef _ALCNN_HPP_
#define _ALCNN_HPP_

#include "ipts_net.hpp"
#include "ipbbox_net.hpp"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
using namespace excalibur;

namespace glasssix
{
	class alcnn
	{
		ipbbox_net* ipbbox;
		ipts_net* ipts;
		//
		int device_;
		std::shared_ptr<tensor> tensor_data = nullptr;
		//
#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
#endif
#ifdef USE_OPENCV
		cv::Mat safetycut(cv::Mat ori, cv::Rect rect);
		void alignface_opencv(cv::Mat& img, cv::Mat& aligned);
#endif
		//prepare for public
#ifdef USE_OPENCV
		std::shared_ptr<tensor> alignface(cv::Mat& img);
#else
		void alignface(std::shared_ptr<tensor>& img, std::shared_ptr<tensor>& aligned);
#endif
	public:
		alcnn(int device);
		~alcnn();

	};
}

#endif //_ALCNN_HPP_