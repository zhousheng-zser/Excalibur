#include "../../include/Longinus/LonginusDetector.hpp"
#include "../../include/Cassius/CassiusFeature.hpp"
#include "../../include/Irisvian/IrisvianSearch.hpp"
#include "opencv2/opencv.hpp"
#include <glasssix\filehelper.hpp>
#include <chrono>

using namespace glasssix;
using namespace glasssix::cassius;

void show_detection_results(cv::Mat img, std::vector<longinus::FaceRectwithFaceInfo> face_info)
{
	for (int i = 0; i < face_info.size(); i++)
	{
		cv::rectangle(img, cv::Rect(face_info[i].x, face_info[i].y, face_info[i].width, face_info[i].height), cv::Scalar(255, 255, 255), 1);
		for (int j = 0; j < 5; j++)
		{
			cv::circle(img, cv::Point(face_info[i].pts[j].x, face_info[i].pts[j].y), 2, cv::Scalar(0, 0, 255), 2);
		}
	}
	cv::imshow("detection results", img);
	cv::waitKey();
}

void show_alignment_results(std::vector<unsigned char> alignedfaces_data, int face_count)
{
	auto face_mats = longinus::encode2mats(alignedfaces_data, face_count);
	for (int i = 0; i < face_count; i++)
	{
		cv::imshow("aligned face", face_mats[i]);
		cv::waitKey();
	}
}

void testCassiusOnly(int device, int order)
{
	cv::Mat src_image = cv::imread("../../images/yswinfread.jpg");
	cassius::CassiusFeature feat_extractor(device);
	//cv::resize(src_image, src_image, cv::Size(128, 128));

	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	auto features = feat_extractor.Forward(src_image.data, 1, order);
	for (size_t i = 0; i < 1; i++)
	{
		auto features = feat_extractor.Forward(src_image.data, 1, order);
		std::cout << "Repeat " << i << "-th times\r";
	}
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	std::cout << "total forward time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 / 10 / 20 << "ms" << std::endl << std::endl;

	std::cout << "size of features:" << features.size() << std::endl;
	std::cout << "size of dimension:" << features[0].size() << std::endl;
	for (int i = 0; i<features.size(); i++)
	{
		for (size_t j = 0; j < 10; j++)
		{
			std::cout << features[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void testRomanciaCassius(int device, int order)
{
	longinus::LonginusDetector detector;
	detector.set(longinus::MULTIVIEW_REINFORCE, device);
	cassius::CassiusFeature feat_extractor(device);
	cv::Mat img = cv::imread("../../images/exciting.png");
	// detection step
	cv::Mat gray;
	cv::cvtColor(img, gray, CV_RGB2GRAY);
	auto face_info = detector.detect(gray.data, gray.cols, gray.rows, gray.step[0], 24, 1.1f, 3, order);
	//show_detection_results(img, face_info);
	// alignment step
	std::vector<std::vector<int>> bboxes;
	std::vector<std::vector<int>> landmarks;
	longinus::extract_faceinfo(face_info, bboxes, landmarks);
	auto alignedfaces_data = detector.alignFace(gray.data, face_info.size(), 1, gray.rows, gray.cols, bboxes, landmarks);
	//show_alignment_results(alignedfaces_data, face_info.size());
	// feature extraction step
	auto features = feat_extractor.Forward(alignedfaces_data.data(), face_info.size(), order);
	//std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	//for (size_t i = 0; i < 100; i++)
	//{
	//	auto features = feat_extractor.Forward(alignedfaces_data.data(), face_info.size(), 0);
	//	std::cout << "Repeat " << i << "-th times\r";
	//}
	//std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	//std::cout << "total forward time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 / 100 << "ms" << std::endl << std::endl;
}

int main()
{
	int device = -1;//device<0(CPU),others(GPU)
	int order = 10;//order==0(NCHW),others(NHWC)
	//testCassiusOnly(device, order);
	//testRomanciaCassius(device, order);
	longinus::LonginusDetector detector;
	detector.set(longinus::MULTIVIEW_REINFORCE, device);
	auto files = glasssix::getFilesinDirectory("C:\\Users\\Glasssix-Admin\\Desktop\\lightingtest\\infread");
	for (int i = 0; i < files.size(); i++)
	{
		cv::Mat img = cv::imread("C:\\Users\\Glasssix-Admin\\Desktop\\lightingtest\\infread\\" + files[i]);
		//(gray.data, gray.cols, gray.rows, gray.step[0], 24, 1.1f, 3, order)
		float thresholds[3] = { 0.6f, 0.7f, 0.7f };
		auto face_info = detector.detectEx(img.data, 3, img.rows, img.cols, 24, thresholds, 0.707, 3);
		//show_detection_results(img, face_info);
	}
	cv::destroyAllWindows();
	system("pause");
	return 0;
}