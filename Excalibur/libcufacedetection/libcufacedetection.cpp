#include "libcufacedetection.hpp"
#include "../Romancia/mtcnn_wrapper.hpp"
#include "../Romancia/npd_wrapper.hpp"
#include <facedetect-dll.h>

namespace glasssix
{
	std::vector<mtcnn_warpper*> mtcs_;
	std::vector<npd_wrapper*> npds_;
	std::vector<unsigned char*> pBuffers_;

	culibface::culibface()
	{
	}

	culibface::~culibface()
	{
		for (auto mtc : mtcs_) 
		{
			try {
				delete mtc;
			}
			catch (...) {

			}
		}
		for (auto npd : npds_)
		{
			try {
				delete npd;
			}
			catch (...) {

			}
		}
		for (auto pBuffer : pBuffers_)
		{
			try {
				delete pBuffer;
			}
			catch (...) {

			}
		}
	}

	int culibface::AddNet(int device)
	{
		auto new_mtc = new mtcnn_warpper(device);
		auto new_npd = new npd_wrapper(device);
		auto new_pBuffer = new unsigned char[0x20000];
		mtcs_.push_back(new_mtc);
		npds_.push_back(new_npd);
		pBuffers_.push_back(new_pBuffer);
		return (mtcs_.size() + npds_.size() + pBuffers_.size()) / 3 - 1;
	}

	std::vector<FaceInfo> culibface::Facedetect_Frontal(const cv::Mat& gray, int min_size, float scale, int model_id)
	{
		std::vector<FaceInfo> output;
		int *pResults = facedetect_frontal(pBuffers_[model_id], (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
			scale, 3, min_size, 0, 0);
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];
			output.push_back(FaceInfo(neighbors, cv::Rect(x, y, w, h)));
		}
		return output;
	}

	std::vector<FaceInfo> culibface::Facedetect_Frontal_Surveillance(const cv::Mat& gray, int min_size, float scale, int model_id)
	{
		std::vector<FaceInfo> output;
		int *pResults = facedetect_frontal_surveillance(pBuffers_[model_id], (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
			scale, 3, min_size, 0, 0);
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];
			output.push_back(FaceInfo(neighbors, cv::Rect(x, y, w, h)));
		}
		return output;
	}

	std::vector<FaceInfo> culibface::Facedetect_Multiview(const cv::Mat& gray, int min_size, float scale, int model_id)
	{
		std::vector<FaceInfo> output;
		int *pResults = facedetect_multiview(pBuffers_[model_id], (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
			scale, 3, min_size, 0, 0);
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];
			output.push_back(FaceInfo(neighbors, cv::Rect(x, y, w, h)));
		}
		return output;
	}

	std::vector<FaceInfo> culibface::Facedetect_Multiview_Reinforce(const cv::Mat& gray, int min_size, float scale, int model_id)
	{
		std::vector<FaceInfo> output;
		int *pResults = facedetect_multiview_reinforce(pBuffers_[model_id], (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
			scale, 3, min_size, 0, 0);
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];
			output.push_back(FaceInfo(neighbors, cv::Rect(x, y, w, h)));
		}
		return output;
	}

	std::vector<FaceInfo> culibface::Facedetect_Multiview_CNN(const cv::Mat& image, int min_size, float scale, int model_id)
	{
		std::vector<FaceInfo> output;
		std::vector<FaceInfoX> infos = mtcs_[model_id]->facedetect_mtcnn(image, min_size);
		for (int i = 0; i < infos.size(); i++)
		{
			int x = (int)infos[i].bbox.xmin;
			int y = (int)infos[i].bbox.ymin;
			int w = (int)(infos[i].bbox.xmax - infos[i].bbox.xmin + 1);
			int h = (int)(infos[i].bbox.ymax - infos[i].bbox.ymin + 1);
			float score = infos[i].bbox.score;
			x = x - (h - w) / 2;
			h = h * 0.85f;
			w = h;
			y = y + 0.2 * h;
			
			float lefteye_nose = abs(infos[i].landmark[0] - infos[i].landmark[4]);
			float righteye_nose = abs(infos[i].landmark[1] - infos[i].landmark[4]);
			float arc_yaw = atan(lefteye_nose / righteye_nose / 10);
			float radius_yaw = arc_yaw / 3.1415 * 180;
			x += w * radius_yaw * 0.01;
			output.push_back(FaceInfo(score, cv::Rect(x, y, w, h)));
		}
		return output;
	}

	std::vector<FaceInfo> culibface::Facedetect_Frontal_Reinforce(const cv::Mat& gray, int min_size, float scale, int model_id)
	{
		std::vector<FaceInfo> output;
		int n = npds_[model_id]->facedetect_npd(gray, min_size);
		auto X = npds_[model_id]->get_x();
		auto Y = npds_[model_id]->get_y();
		auto S = npds_[model_id]->get_size();
		auto scores = npds_[model_id]->get_score();
		for (int i = 0; i < n; i++)
		{
			output.push_back(FaceInfo(scores[i], cv::Rect(X[i], Y[i], S[i], S[i])));
		}
		return output;
	}

}