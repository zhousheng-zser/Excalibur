#include <iostream>
#include <longinus_c.h>
#include <opencv2/opencv.hpp>

int main()
{
	auto d = Longinus_NewInstance(-1);
	cv::Mat img = cv::imread("F:/A1010.jpg");
	std::cout << "img.channels(): " << img.channels() << std::endl;
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	glasssix::longinus::face_rect_with_face_info* info = nullptr;
	//int n = Longinus_detectRetina(d, &info, img.data, 48, img.rows, img.cols, 1, 0.5f);

	float aaa[3] = { 0.6, 0.7, 0.7 };
	int n = Longinus_detectEx(d, &info, img.data, img.rows, img.cols, 48, aaa, 1.0 / 0.709f, 3, 1);
	std::cout << "n:" << n << std::endl;
	//unsigned char image[640*480*1]={};
	if (n <= 0)
		return 0;
	std::vector<int> bbox = { info[0].x, info[0].y, info[0].width, info[0].height };
	std::cout << info[0].x << " " << info[0].y << " " << info[0].width << " " << info[0].height << std::endl;
	std::vector<int> landmark;
	for (int i = 0; i < 5; i++)
	{
		landmark.push_back(info->pts[i].x);
		landmark.push_back(info->pts[i].y);
	}
	//int landmarks[10]{125,125,175,135,150,150,175,125,175,165};
	Longinus_alignFace(d, gray.data, 1, img.rows, img.cols, bbox.data(), landmark.data());
	std::cout << "=============================" << std::endl;
}
