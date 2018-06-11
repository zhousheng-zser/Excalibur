#ifndef _ALIGNMENT_HPP_
#define _ALIGNMENT_HPP_

#include "ipts_net.hpp"
#include "ipbbox_net.hpp"

#ifdef USE_OPENCV
#include <opencv2\opencv.hpp>
#endif
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

#ifdef USE_OPENCV
		cv::Mat saftycut(cv::Mat ori_image, cv::Rect roi);
		static cv::Rect calc_margin_rect(cv::Rect& face_rect, cv::Point2f& center, float roll, float pitch);
#endif // USE_OPENCV

		static rectangle<int> calc_margin_rect(rectangle<int>& face_rect, point<float>& center, float roll, float pitch);

		std::vector<int> yaw;
		std::vector<int> pitch;
		std::vector<int> roll;
	public:
		alignment(int device);
		~alignment();
#ifdef USE_OPENCV
		void alignment_face(cv::Mat& img, cv::Mat& aligned);
		void alignment_face(std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& aligned_faces);
		std::vector<std::vector<float>> F_Ipbbox(std::vector<cv::Mat> imgs);
		std::vector<float> F_Ipts(std::vector<cv::Mat> imgs);
#endif // USE_OPENCV
		std::vector<int> get_yaw_angle()
		{
			return yaw;
		}
		std::vector<int> get_pitch_angle()
		{
			return pitch;
		}
		std::vector<int> get_roll_angle()
		{
			return roll;
		}
	};
}
#endif // !_ALIGNMENT_HPP_
