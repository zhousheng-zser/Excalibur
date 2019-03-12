#include "../../include/Longinus/LonginusDetector.hpp"
#include "../../include/Cassius/CassiusFeature.hpp"
#include "../../include/Irisvian/IrisvianSearch.hpp"
#include "opencv2/opencv.hpp"

using namespace glasssix;

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

int main()
{
	longinus::LonginusDetector detector;
	detector.set(longinus::MULTIVIEW_REINFORCE, -1);
	cassius::CassiusFeature feat_extractor(-1);
	cv::Mat img = cv::imread("../../images/exciting.png");
	// detection step
	cv::Mat gray;
	cv::cvtColor(img, gray, CV_RGB2GRAY);
	auto face_info = detector.detect(gray.data, gray.cols, gray.rows, gray.step[0], 24, 1.1f, 3, 0);
	show_detection_results(img, face_info);
	// alignment step
	std::vector<std::vector<int>> bboxes;
	std::vector<std::vector<int>> landmarks;
	longinus::extract_faceinfo(face_info, bboxes, landmarks);
	auto alignedfaces_data = detector.alignFace(gray.data, face_info.size(), 1, gray.rows, gray.cols, bboxes, landmarks);
	show_alignment_results(alignedfaces_data, face_info.size());
	// feature extraction step
	auto features = feat_extractor.Forward(alignedfaces_data.data(), face_info.size(), 0);
	return 0;
}