#include "nplogic.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <facedetect-dll.h>
using namespace glasssix;

void detect_thread(nplogic* npd, cv::Mat gray, int id)
{
	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	for (size_t i = 0; i < 1000; i++)
	{
		npd->detect(gray.data, gray.cols, gray.rows, 48);
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	std::cout << "Thread " << id << ", " << "total detection time:" 
		<< (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 / 1000 - 5<< "ms" << std::endl;
}

void multi_thread_test()
{
	std::vector<nplogic*> npds;
	std::vector<std::thread> ts;
	std::vector<cv::Mat> grays;
	for (size_t i = 0; i < 8; i++)
	{
		grays.push_back(cv::imread("C:\\Users\\BALTHASAR\\Desktop\\keliamoniz1.jpg", CV_LOAD_IMAGE_GRAYSCALE));
		nplogic* npd = new nplogic(i%2);
		npds.push_back(npd);
		npds[i]->load("D:\\Research\\CudaNpdDetect\\model\\result_1223_1.bin");
		ts.push_back(std::thread(detect_thread, npds[i], grays[i], i));
	}
	for (size_t i = 0; i < 8; i++)
	{
		ts[i].detach();
	}
}

void detect_thread_libface(cv::Mat gray, int id)
{
	unsigned char * pBuffer = (unsigned char *)malloc(0x20000);
	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	for (size_t i = 0; i < 1000; i++)
	{
		facedetect_frontal(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
			1.2f, 2, 48, 0, 1);
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	std::cout << "Thread " << id << ", " << "total detection time:" 
		<< (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 / 1000 - 5<< "ms" << std::endl;
}

void multi_thread_test_libface()
{
	std::vector<std::thread> ts;
	std::vector<cv::Mat> grays;
	for (size_t i = 0; i < 8; i++)
	{
		grays.push_back(cv::imread("C:\\Users\\BALTHASAR\\Desktop\\keliamoniz1.jpg", CV_LOAD_IMAGE_GRAYSCALE));
		ts.push_back(std::thread(detect_thread_libface, grays[i], i));
	}
	for (size_t i = 0; i < 8; i++)
	{
		ts[i].detach();
	}
}

int main()
{
	multi_thread_test();
	//multi_thread_test_libface();
	std::this_thread::sleep_for(std::chrono::seconds(300));
	nplogic* npd = new nplogic(0);
	//npd->load();
	npd->load("D:\\Research\\CudaNpdDetect\\model\\result_1223_1.bin");
	//cv::Mat gray = cv::imread("E:\\Data\\LS3D-W\\300W-Testset-3D\\outdoor_237.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat gray = cv::imread("C:\\Users\\BALTHASAR\\Desktop\\keliamoniz1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int n = npd->detect(gray.data, gray.cols, gray.rows, 48);
	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	for (int i = 0; i < 5000; i++)
	{
		npd->detect(gray.data, gray.cols, gray.rows, 48);
	}
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	std::cout << "total detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 / 5000 << "ms" << std::endl << std::endl;
	std::vector< int >& Xs = npd->getXs();
	std::vector< int >& Ys = npd->getYs();
	std::vector< int >& Ss = npd->getSs();
	std::vector< float >& Scores = npd->getScores();

	std::vector< cv::Rect> facesRect;
	std::vector< float > facesScore;
	for (int i = 0; i < n; i++)
	{
		if (Scores[i] >= 5)
		{
			cv::Rect rect(Xs[i], Ys[i], Ss[i], Ss[i]);
			cv::rectangle(gray, rect, cv::Scalar(0, 0, 255), 2);
			std::cout << "Face " << i << ": " << Xs[i] << ", " << Ys[i] << ", " << Ss[i] << ", " << Ss[i] << "\t " << Scores[i] << std::endl;
		}
	}
	while (gray.rows >= 1080)
	{
		cv::resize(gray, gray, cv::Size(gray.cols * 0.717, gray.rows*0.717));
	}
	cv::imshow("test", gray);
	cv::waitKey();
	return 0;
}