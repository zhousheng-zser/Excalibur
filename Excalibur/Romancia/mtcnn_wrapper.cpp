#include "mtcnn_wrapper.hpp"

namespace glasssix
{
	mtcnn_warpper::mtcnn_warpper(int device)
	{
		device_ = device;
		mt_ = new MTCNN(device_);
	}

	mtcnn_warpper::~mtcnn_warpper()
	{
		delete mt_;
	}

	std::vector<FaceInfoX> mtcnn_warpper::facedetect_mtcnn(unsigned char* image_data, int width, int height, int min_size)
	{
		cv::Mat image = cv::Mat(height, width, CV_8UC3, image_data);
		return mt_->Detect(image, min_size, threshold, factor, 3);
	}
}