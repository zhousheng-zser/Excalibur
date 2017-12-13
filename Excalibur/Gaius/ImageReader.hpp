#pragma once
#ifndef _IMAGEREADER_HPP_
#define _IMAGEREADER_HPP_

#include "ImageTensor.hpp"

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace excalibur
{
	class ImageReader
	{
	public:
		ImageReader();
		~ImageReader();
#ifdef USE_OPENCV
		static void images2tensor(const std::vector<cv::Mat> images, std::shared_ptr<ImageTensor<unsigned char>>& tensor_data);

		static void image2tensor(const cv::Mat image, std::shared_ptr<ImageTensor<unsigned char>>& tensor_data);

		static void tensor2image(const std::shared_ptr<ImageTensor<unsigned char>> tensor_data, cv::Mat& image);
#endif
	};
}

#endif // _IMAGEREADER_HPP_