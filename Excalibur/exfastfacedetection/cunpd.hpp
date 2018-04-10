#ifndef _CUNPD_HPP_
#define _CUNPD_HPP_

#ifdef CUNPD_EXPORTS
#define CUNPD_DLL __declspec(dllexport)
#else
#define CUNPD_DLL __declspec(dllimport)
#endif

#include <opencv2\opencv.hpp>
#include <string>
#include <vector>

namespace glasssix
{
	struct FaceInfomation
	{
		cv::Rect rect;
		float score;
	};

	class CUNPD_DLL cunpd
	{
	public:
		cunpd() {};
		int AddNpdModel(int device);
		int AddNpdModel(std::string modelpath, int device);
		std::vector<FaceInfomation> detect(cv::Mat img, int model_id, int min_size);
		~cunpd();
	};
}

#endif