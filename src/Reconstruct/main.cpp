#include "opencv2/opencv.hpp"
#include "deconv_net.hpp"
#include <glasssix/tensor.hpp>

using namespace glasssix::excalibur;
using namespace glasssix::longinus;

void testDeconv(int device, int order)
{
	//cv::Mat src_image = cv::imread("../../images/yswinfread.jpg");
	cv::Mat src_image = cv::imread("D:/12.jpg");
	cv::resize(src_image, src_image, cv::Size(102, 126));
	cv::Mat gray;
	cv::cvtColor(src_image, gray, CV_RGB2GRAY);
	cv::imshow("44", gray);
	//cv::waitKey(0);

	deconvT testor(device);
	auto res = testor.Forward(gray.data,1, order);
	
	for (int i = 0; i < res.size(); i++)
	{
		cv::Mat dd(364, 292, CV_8UC1, res[i].data());
		cv::imshow("dd", dd);
		cv::waitKey(0);
	}
}

int main()
{
	int device = -1;//device<0(CPU),others(GPU)
	int order = 0;//order==0(NCHW),others(NHWC)

	testDeconv(device, order);

	cv::destroyAllWindows();
	//system("pause");
	return 0;
}