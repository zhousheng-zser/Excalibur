#ifndef _ALIGNMENT_HPP_
#define _ALIGNMENT_HPP_

#include "ipts_net.hpp"
#include "ipbbox_net.hpp"
#include <opencv2\opencv.hpp>

using namespace excalibur;

namespace glasssix
{
	class alignment
	{
		std::shared_ptr<ipbbox_net> ipbbox;
		std::shared_ptr<ipts_net> ipts;
		//
		int device_;
		std::shared_ptr<tensor<float>> tensor_data = nullptr;
		//
#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
#endif
		cv::Mat saftycut(cv::Mat ori_image, cv::Rect roi);

	public:
		alignment(int device);
		~alignment();

		void alignment_face(cv::Mat& img, cv::Mat& aligned);
	};
}
#endif // !_ALIGNMENT_HPP_
