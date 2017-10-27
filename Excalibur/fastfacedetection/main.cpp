#include <iostream>
#include "../Excalibur/io.hpp"
#include "mtcnn.hpp"
#include <filesystem>

using namespace excalibur;
using namespace fastface;

void unittest()
{
	cv::Mat image = cv::imread("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\re_12_18.jpg");
	cv::Mat image_ = cv::imread("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\re_12_18.jpg");
	//cv::Mat image = cv::imread("E:\\datasets\\LS3D-W\\300W-Testset-3D\\indoor_087_0.png");
	/*cv::resize(image, image, cv::Size(12, 18));
	cv::imwrite("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\re_12_18.jpg", image);*/
	std::shared_ptr<tensor> tensor_data = nullptr;
	std::shared_ptr<tensor> tensor_data_ = nullptr;
	//io::image2tensor(image, tensor_data/*, false, 1.0*/);
	io::images2tensor(std::vector<cv::Mat>{image}, tensor_data);
	io::images2tensor(std::vector<cv::Mat>{image_}, tensor_data_);
	mtcnn_pnet pnet = mtcnn_pnet(-1);
	//mtcnn_rnet rnet = mtcnn_rnet();
	//mtcnn_onet onet = mtcnn_onet();

	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	for (int i = 0; i < 1; i++)
	{
		if (i%2==0)
		{
			pnet.Forward(tensor_data);
		}
		else
		{
			pnet.Forward(tensor_data_);
		}
		//std::cout << std::endl;
	}
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	std::cout << "total forward time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl << std::endl;
	auto a = pnet.get_prob1();
	const float* output = a->cpu_data();
	for (int i = 0; i<a->count()/1 ; i++)
	{
		std::cout << output[i] << " ";
		if (i == a->count() / 2 - 1)
		{
			std::cout << std::endl;
		}
	}
}

void mtcnntset()
{
	cv::Mat image = cv::imread("C:\\Users\\bj12\\Desktop\\WeChat Image_20171026153606.jpg");
	cv::resize(image, image, cv::Size(750, 1000));
	mtcnn mt = mtcnn();
	double threshold[3] = { 0.7, 0.7, 0.7 };
	double factor = 0.709;
	int minSize = 48;
	std::vector<FaceInfo> faceInfo;
	mt.Detect(image, faceInfo, minSize, threshold, factor);
	mtcnn::drawDectionResult(image, faceInfo);
	imshow("final", image);
	cv::waitKey(0);
}

int main()
{
	//unittest();
	mtcnntset();
	return 0;
}