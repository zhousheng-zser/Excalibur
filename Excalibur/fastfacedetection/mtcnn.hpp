#pragma once
#ifndef _MTCNN_HPP_
#define _MTCNN_HPP_
#include "mtcnn_pnet.hpp"
#include "mtcnn_onet.hpp"
#include "mtcnn_rnet.hpp"

namespace fastface
{
	typedef struct FaceRect {
		float x1;
		float y1;
		float x2;
		float y2;
		float score; /**< Larger score should mean higher confidence. */
	} FaceRect;

	typedef struct FacePts {
		float x[5], y[5];
	} FacePts;

	typedef struct FaceInfo {
		FaceRect bbox;
		cv::Vec4f regression;
		FacePts facePts;
		double roll;
		double pitch;
		double yaw;
	} FaceInfo;
	class mtcnn {
	public:
		mtcnn(int device);
		void Detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo, int minSize, double* threshold, double factor);
		static void drawDectionResult(cv::Mat &frame, std::vector<FaceInfo> &faceInfo);
	private:
		void GenerateBoundingBox(std::shared_ptr<tensor> confidence, std::shared_ptr<tensor> reg,
			float scale, float thresh, int image_width, int image_height);
		void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_single,
			int net_id, double thresh, char netName);
		std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
		void Bbox2Square(std::vector<FaceInfo>& bboxes);
		void Padding(int img_w, int img_h);
		std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);

	private:
		mtcnn_pnet *PNet_;
		mtcnn_rnet *RNet_;
		mtcnn_onet *ONet_;

		// x1,y1,x2,t2 and score
		std::vector<FaceInfo> condidate_rects_;
		std::vector<FaceInfo> total_boxes_;
		std::vector<FaceInfo> regressed_rects_;
		std::vector<FaceInfo> regressed_pading_;

		std::vector<cv::Mat> crop_img_;
		int curr_feature_map_w_;
		int curr_feature_map_h_;
		int num_channels_;

		int device_;
	};
}

#endif