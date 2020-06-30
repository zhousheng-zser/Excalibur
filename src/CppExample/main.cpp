#include <iostream>
#include <opencv2/opencv.hpp>
#include "../../include/Primitives/profiler.hpp"
#include "retina_face.hpp"

using namespace glasssix;

int main()
{
	retina_face *rf = new retina_face("D:\\Research\\Excalibur\\models\\retina.phai", "D:\\Research\\Excalibur\\models\\retina.racy");
	
	cv::Mat img = cv::imread("C:\\Users\\Glasssix-Admin\\Desktop\\480p.jpg");
	//pro->turn_on();
	cv::Mat small_img;
	cv::resize(img, small_img, cv::Size(img.cols / 2, img.rows / 2));
	auto face_info = rf->detect(small_img, 0.3);
	for (size_t i = 0; i < 5; i++)
	{
		rf->detect(small_img, 0.3);
	}
	/*profiler* pro = profiler::get();
	pro->turn_off();
	pro->dump_profile("D:\\Research\\Excalibur\\models\\retina_480.json");*/
	for (size_t i = 0; i < face_info.size(); i++)
	{
		cv::rectangle(img, cv::Rect(face_info[i].rect.x1, face_info[i].rect.y1, (face_info[i].rect.x2 - face_info[i].rect.x1), (face_info[i].rect.y2 - face_info[i].rect.y1)), cv::Scalar(0, 255, 0));
	}
	cv::imshow("img", img);
	cv::waitKey();
	return 0;
}
