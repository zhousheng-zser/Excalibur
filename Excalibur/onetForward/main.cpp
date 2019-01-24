
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "D:/projects/MCL_Forward/Excalibur/Excalibur/tensor_operation_cpu.hpp"
#include "D:/projects/MCL_Forward/Excalibur/Excalibur/tensor_utils.hpp"
#include "onetForward.hpp"
#include "onetForwardData.hpp"
#include "RomanciaDetector.hpp"

using namespace cv;
using namespace std;
using namespace excalibur;
using namespace glasssix;
using namespace glasssix;
using namespace glasssix::longinus;
using namespace std;


static cv::Mat saftycut(cv::Mat ori_image, cv::Rect roi)
{
	cv::Mat output = cv::Mat(roi.height, roi.width, ori_image.type());
	if (roi.x >= 0 && roi.y >= 0 && (roi.x + roi.width <= ori_image.cols) && (roi.y + roi.height <= ori_image.rows))
	{
		ori_image(roi).copyTo(output);
	}
	else
	{
		int top = std::max(0, -1 * roi.y);
		int bottom = std::max(roi.y + roi.height - ori_image.rows, 0);
		int left = std::max(0, -1 * roi.x);
		int right = std::max(roi.x + roi.width - ori_image.cols, 0);
		cv::Mat temp_origin_with_border;
		cv::copyMakeBorder(ori_image, temp_origin_with_border, top, bottom, left, right, cv::BORDER_CONSTANT);
		roi.x += left;
		roi.y += top;
		temp_origin_with_border(roi).copyTo(output);
	}
	return output;
}

int main()
{
	//输入图片并转为灰度图
	Mat src_image, gray_image;
	src_image = imread("D:/projects/MCL_Forward/Excalibur/onetForward/print/12.jpg");
	int height = src_image.rows;
	int width = src_image.cols;
	cvtColor(src_image, gray_image, CV_BGR2GRAY);

	//检测并绘制人脸框
	RomanciaDetector detector;
	detector.set(MULTIVIEW_REINFORCE, -1);
	std::vector<FaceRect> faceBox;
	faceBox = detector.detect(gray_image.data, width, height, 80, 1.1f, 3, false, false);
	std::cout << "NUM:" << faceBox.size() << std::endl;
	for (size_t i = 0; i < faceBox.size(); i++)
	{
		cv::rectangle(src_image, cv::Rect(faceBox[i].x, faceBox[i].y, faceBox[i].width, faceBox[i].height), cv::Scalar(255, 0, 0), 1);
	}

	//在灰度图上截取人脸框，并缩放至48×48
	cv::Mat ROI = saftycut(gray_image, cv::Rect(faceBox[0].x, faceBox[0].y, faceBox[0].width, faceBox[0].height));
	cv::resize(ROI, ROI, cv::Size(48, 48));

	//灰度图人脸框的数据拷贝至tensor<unsigned char>
	tensor<unsigned char> ROI_tensor;
	shared_ptr<tensor<unsigned char>> ROI_ptr = make_shared<tensor<unsigned char>>(ROI_tensor);
	ROI_ptr.reset(new tensor<unsigned char>(std::vector<int>{1, 1, 48, 48}, -1, NCHW));
	tensor_operation_cpu::mat2tensor(ROI, ROI_ptr);

	//tensor<unsigned char>转换格式为tensor<float>
	tensor<float> ROI_tensor2;
	shared_ptr<tensor<float>> ROI_ptr2 = make_shared<tensor<float>>(ROI_tensor2);
	ROI_ptr2.reset(new tensor<float>(std::vector<int>{1, 1, 48, 48}, -1, NCHW));
	tensor_operation_cpu::type_convertor_cpu(ROI_ptr, ROI_ptr2);

	//forward，并取出关键点参数
	onetForward onet(0);
	onet.Forward(ROI_ptr2);
	std::shared_ptr<tensor<float>> param_conv6_3_tensor = onet.get_conv6_3();
	const float* param_conv6_3_data = param_conv6_3_tensor->cpu_data();

	//通过关键点参数和人脸框尺寸，计算关键点位置坐标
	float xLeftEye = faceBox[0].x + param_conv6_3_data[0] * faceBox[0].width;
	float yLeftEye = faceBox[0].y + param_conv6_3_data[1] * faceBox[0].height;
	float xRightEye = faceBox[0].x + param_conv6_3_data[2] * faceBox[0].width;
	float yRightEye = faceBox[0].y + param_conv6_3_data[3] * faceBox[0].height;
	float xNose = faceBox[0].x + param_conv6_3_data[4] * faceBox[0].width;
	float yNose = faceBox[0].y + param_conv6_3_data[5] * faceBox[0].height;
	float xLeftMouth = faceBox[0].x + param_conv6_3_data[6] * faceBox[0].width;
	float yLeftMouth = faceBox[0].y + param_conv6_3_data[7] * faceBox[0].height;
	float xRightMouth = faceBox[0].x + param_conv6_3_data[8] * faceBox[0].width;
	float yRightMouth = faceBox[0].y + param_conv6_3_data[9] * faceBox[0].height;

	//绘制、显示关键点
	circle(src_image, Point2f(xLeftEye, yLeftEye), 3, Scalar(0, 255, 0), -1);
	circle(src_image, Point2f(xRightEye, yRightEye), 3, Scalar(0, 255, 0), -1);
	circle(src_image, Point2f(xNose, yNose), 3, Scalar(0, 255, 0), -1);
	circle(src_image, Point2f(xLeftMouth, yLeftMouth), 3, Scalar(0, 255, 0), -1);
	circle(src_image, Point2f(xRightMouth, yRightMouth), 3, Scalar(0, 255, 0), -1);
	imshow("src", src_image);
	waitKey(0);

	system("pause");
	return 0;
}