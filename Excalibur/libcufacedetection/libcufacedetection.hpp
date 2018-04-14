#ifndef _LIBCUFACEDETECTION_HPP_
#define _LIBCUFACEDETECTION_HPP_

#ifdef LIBCUFACEDETECTION_EXPORTS
#define FACEDETECTION_DLL __declspec(dllexport)
#else
#define FACEDETECTION_DLL __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace glasssix
{
	struct FaceInfo {
		float scores;
		cv::Rect rects;
		FaceInfo(float scores_, cv::Rect rects_) :scores(scores_), rects(rects_) {};
	};

	class FACEDETECTION_DLL culibface
	{
	public:
		culibface();
		~culibface();
		int AddNet(int device = -1);

		std::vector<FaceInfo> Facedetect_Frontal(const cv::Mat &gray, int min_size, float scale, int model_id);
		std::vector<FaceInfo> Facedetect_Multiview(const cv::Mat &gray, int min_size, float scale, int model_id);
		std::vector<FaceInfo> Facedetect_Multiview_Reinforce(const cv::Mat &gray, int min_size, float scale, int model_id);
		std::vector<FaceInfo> Facedetect_Frontal_Surveillance(const cv::Mat &gray, int min_size, float scale, int model_id);
		std::vector<FaceInfo> Facedetect_Multiview_CNN(const cv::Mat &image, int min_size, float scale, int model_id);
		std::vector<FaceInfo> Facedetect_Frontal_Reinforce(const cv::Mat &gray, int min_size, float scale, int model_id);
	private:
	};

	
}

#endif // !_LIBCUFACEDETECTION_HPP_
