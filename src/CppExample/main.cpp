#include <iostream>
#include <opencv2/opencv.hpp>
#include "../../include/Primitives/profiler.hpp"
#include "retina_face.hpp"

using namespace glasssix;

int main()
{
	std::shared_ptr<excalibur::pipeline<float>> p;
	p.reset(new excalibur::pipeline<float>("D:\\Research\\DAG\\exdag\\unicorn.phai", "D:\\Research\\DAG\\exdag\\unicorn.racy"));
	retina_face *rf = new retina_face("D:\\Research\\Excalibur\\models\\retina.phai", "D:\\Research\\Excalibur\\models\\retina.racy");
	
	cv::Mat img = cv::imread("C:\\Users\\Glasssix-Admin\\Desktop\\480p2.jpg");
	//pro->turn_on();
	/*cv::Mat small_img;
	cv::resize(img, small_img, cv::Size(img.cols / 2, img.rows / 2));*/
	auto face_info = rf->detect(img, 0.3);
	/*for (size_t i = 0; i < 5; i++)
	{
		rf->detect(img, 0.3);
	}*/
	/*profiler* pro = profiler::get();
	pro->turn_off();
	pro->dump_profile("D:\\Research\\Excalibur\\models\\retina_480.json");*/
	for (size_t i = 0; i < face_info.size(); i++)
	{
		cv::rectangle(img, cv::Rect(face_info[i].rect.x1, face_info[i].rect.y1, (face_info[i].rect.x2 - face_info[i].rect.x1), (face_info[i].rect.y2 - face_info[i].rect.y1)), cv::Scalar(0, 255, 0));
		for (size_t j = 0; j < 5; j++)
		{
			cv::circle(img, cv::Point(face_info[i].pts.x[j], face_info[i].pts.y[j]), 2, cv::Scalar(0, 0, 255), 2);
		}
	}
	cv::imshow("img", img);
	cv::waitKey();
	return 0;
}
