#include <iostream>
#include "../Excalibur/io.hpp"
#include "Damocles.hpp"
#include <opencv2/opencv.hpp>


using namespace glasssix;
using namespace glasssix::excalibur;
using namespace glasssix::longinus;

//void unittest()
//{
//	cv::Mat image = cv::imread("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\re_12_18.jpg");
//	cv::Mat image_ = cv::imread("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\re_12_18.jpg");
//	//cv::Mat image = cv::imread("E:\\datasets\\LS3D-W\\300W-Testset-3D\\indoor_087_0.png");
//	/*cv::resize(image, image, cv::Size(12, 18));
//	cv::imwrite("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\re_12_18.jpg", image);*/
//	std::shared_ptr<tensor<float>> tensor_data;
//	std::shared_ptr<tensor<float>> tensor_data_ = nullptr;
//	//io::image2tensor(image, tensor_data/*, false, 1.0*/);
//	io::images2tensor(std::vector<cv::Mat>{image}, tensor_data);
//	io::images2tensor(std::vector<cv::Mat>{image_}, tensor_data_);
//	mtcnn_pnet pnet = mtcnn_pnet(-1);
//	//mtcnn_rnet rnet = mtcnn_rnet();
//	//mtcnn_onet onet = mtcnn_onet();
//
//	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
//	for (int i = 0; i < 1; i++)
//	{
//		if (i%2==0)
//		{
//			pnet.Forward(tensor_data);
//		}
//		else
//		{
//			pnet.Forward(tensor_data_);
//		}
//		//std::cout << std::endl;
//	}
//	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
//	std::cout << "total forward time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl << std::endl;
//	auto a = pnet.get_prob1();
//	const float* output = a->cpu_data();
//	for (int i = 0; i<a->count()/1 ; i++)
//	{
//		std::cout << output[i] << " ";
//		if (i == a->count() / 2 - 1)
//		{
//			std::cout << std::endl;
//		}
//	}
//}

void mtcnntset()
{
	cv::Mat image = cv::imread("D:\\3.jpg");
	//C:\\Users\\BALTHASAR\\Desktop\\procesed.jpg  D:\\Detection-Data\\face\\ibug\\image_005_1.jpg
	//cv::resize(image, image, cv::Size(375, 500));


	float factor = 0.709f;
	float threshold[3] = { 0.7f, 0.6f, 0.6f };
	int minSize = 24;
	int channels = image.channels();
	int height = image.rows;
	int width = image.cols;
	std::vector<FaceInfomation> faceInfo;

	
	Damocles detector(-1);
	faceInfo = detector.Detect(image.data, channels, height, width, minSize, threshold, factor, 3, 0);
	

	//double t = (double)cv::getTickCount();
	//int execute_times = 10;
	//for (int i = 0; i < execute_times; i++)
	//{
	//	detector.Detect(image.data, channels, height, width, minSize, threshold, factor, 3);
	//}
	//std::cout << "Detection time: " << (double)(cv::getTickCount() - t) / cv::getTickFrequency() / execute_times * 1000 << "ms" << std::endl;
	for (int i = 0; i < faceInfo.size(); i++) {
		int x = (int)faceInfo[i].bbox.xmin;
		int y = (int)faceInfo[i].bbox.ymin;
		int w = (int)(faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
		int h = (int)(faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
		std::cout << faceInfo[i].bbox.score << std::endl;
		cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
	}
	for (int i = 0; i < faceInfo.size(); i++) {
		float *landmark = faceInfo[i].landmark;
		for (int j = 0; j < 5; j++) {
			cv::circle(image, cv::Point((int)landmark[2 * j], (int)landmark[2 * j + 1]), 1, cv::Scalar(255, 255, 0), 2);
		}
	}

	imshow("final", image);
	cv::waitKey(10);
}

int main()
{
	//unittest();
	mtcnntset();
	system("pause");
	return 0;
}
