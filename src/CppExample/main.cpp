#include <iostream>
#include <opencv2/opencv.hpp>
#include "../../include/Primitives/profiler.hpp"
#include "retina_face.hpp"
#include "ultra_face.hpp"
using namespace glasssix;

int main()
{
	cv::Mat img = cv::imread("C:\\Users\\Glasssix-Admin\\Desktop\\480p2.jpg");
	retina_face *rf = new retina_face("D:\\Research\\Excalibur\\models\\retina.phai", "D:\\Research\\Excalibur\\models\\retina.racy", 0.4, 0);
	/*cv::Mat small_img;
	cv::resize(img, small_img, cv::Size(img.cols / 2, img.rows / 2));*/
	auto face_info = rf->detect(img, 0.3);
	for (size_t i = 0; i < 5; i++)
	{
		rf->detect(img, 0.3);
	}
	for (size_t i = 0; i < face_info.size(); i++)
	{
		cv::rectangle(img, cv::Rect(face_info[i].rect.x1/* * 2*/, face_info[i].rect.y1/* * 2*/, (face_info[i].rect.x2 - face_info[i].rect.x1)/* * 2*/, (face_info[i].rect.y2 - face_info[i].rect.y1)/* * 2*/), cv::Scalar(0, 255, 0));
		for (size_t j = 0; j < 5; j++)
		{
			cv::circle(img, cv::Point(face_info[i].pts.x[j]/* * 2*/, face_info[i].pts.y[j]/* * 2*/), 2, cv::Scalar(0, 0, 255), 2);
		}
	}
	cv::imshow("img", img);
	cv::waitKey();
	return 0;
}
