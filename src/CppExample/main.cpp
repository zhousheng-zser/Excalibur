#include <iostream>
#include <opencv2/opencv.hpp>
#include "retina_face.hpp"

int main()
{
	retina_face *rf = new retina_face(std::string("1"), "net3");

	// cv::VideoCapture cap(1);
	cv::Mat img = cv::imread("C:\\Users\\Glasssix-Admin\\Desktop\\1080p.png");

	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	auto faceInfo = rf->detect(img, 0.3);
	std::vector<std::vector<int>> bboxes;
	std::vector<std::vector<int>>landmarks;
	std::vector<cv::Mat> detected_faces;
	for (size_t i = 0; i < faceInfo.size(); i++)
	{
		cv::rectangle(img, cv::Rect(faceInfo[i].rect.x1, faceInfo[i].rect.y1, (faceInfo[i].rect.x2 - faceInfo[i].rect.x1), (faceInfo[i].rect.y2 - faceInfo[i].rect.y1)), cv::Scalar(0, 255, 0));
	}
	cv::imshow("img", img);
	cv::waitKey();
	return 0;
}
