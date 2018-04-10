#ifndef _NPD_WRAPPER_HPP_
#define _NPD_WRAPPER_HPP_

#include "../exfastfacedetection/nplogic.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace glasssix
{
	class npd_wrapper
	{
	public:
		npd_wrapper(int device);
		~npd_wrapper();

		int facedetect_npd(unsigned char* image_data, int width, int height, int min_size);
		int facedetect_npd(const cv::Mat &gray, int min_size);
		std::vector<int> get_x();
		std::vector<int> get_y();
		std::vector<int> get_size();
		std::vector<float> get_score();

	private:
		nplogic* npd_;
		int device_;
	};
}
#endif