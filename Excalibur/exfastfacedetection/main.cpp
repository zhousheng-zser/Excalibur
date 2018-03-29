#include "nplogic.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace glasssix;

int main()
{
	nplogic* npd = new nplogic(0);
	//npd->load();
	npd->load("D:\\Research\\CudaNpdDetect\\model\\result_1223_1.bin");
	cv::Mat gray = cv::imread("E:\\Data\\LS3D-W\\300W-Testset-3D\\outdoor_237.png", CV_LOAD_IMAGE_GRAYSCALE);
	int n = npd->detect(gray.data, gray.cols, gray.rows, 48);
	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	for (int i = 0; i < 500; i++)
	{
		npd->detect(gray.data, gray.cols, gray.rows, 48);
	}
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	std::cout << "total detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 / 500 << "ms" << std::endl << std::endl;
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