#ifndef _RETINA_FACE_HPP_
#define _RETINA_FACE_HPP_

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "../../include/Excalibur/pipeline.hpp"

namespace glasssix
{
	struct anchor_win
	{
		float x_ctr;
		float y_ctr;
		float w;
		float h;
	};

	struct anchor_box
	{
		float x1;
		float y1;
		float x2;
		float y2;
	};

	struct face_pts
	{
		float x[5];
		float y[5];
	};

	struct face_detect_info
	{
		float score;
		anchor_box rect;
		face_pts pts;
	};

	struct anchor_cfg
	{
	public:
		int STRIDE;
		std::vector<int> SCALES;
		int BASE_SIZE;
		std::vector<float> RATIOS;
		int ALLOWED_BORDER;

		anchor_cfg()
		{
			STRIDE = 0;
			SCALES.clear();
			BASE_SIZE = 0;
			RATIOS.clear();
			ALLOWED_BORDER = 0;
		}
	};

	class retina_face
	{
	public:
		retina_face(std::string phai_path, std::string racy_path, float nms = 0.4, int device = -1);
		~retina_face();

		// Batch process have some advantage in inference but can't speed up preprocess and postprocess
		// TODO: implement
		std::vector<std::vector<face_detect_info>> detectBatchImages(std::vector<cv::Mat> imgs, float threshold = 0.5);
		//Test in GTX1060:
		// | model | speed | input size | preprocess time | inference | postprocess time |
		//	| :------ : | : ---- : | : -------- : | : ------------ - : | : ------ - : | : -------------- : |
		//	|  caffe | ????ms | 1920x1080 | ????ms | 61ms | ????ms      |
		//	|  caffe | ????ms | 1280ｘ720 | ????ms | 44ms | ????ms      |
		//	|  caffe | 17.3ms | 640ｘ480 | 3.9ms | 13.4ms | 1.0ms |
		std::vector<face_detect_info> detect(const cv::Mat img, float threshold = 0.5, float scales = 1.0);
	private:
		std::vector<face_detect_info> postProcess(int inputW, int inputH, float threshold);
		anchor_box bbox_pred(anchor_box anchor, cv::Vec4f regress);
		std::vector<anchor_box> bbox_pred(std::vector<anchor_box> anchors, std::vector<cv::Vec4f> regress);
		std::vector<face_pts> landmark_pred(std::vector<anchor_box> anchors, std::vector<face_pts> face_pts);
		face_pts landmark_pred(anchor_box anchor, face_pts facePt);
		static bool compare_bbox(const face_detect_info &a, const face_detect_info &b);
		std::vector<face_detect_info> nms(std::vector<face_detect_info> &bboxes, float threshold);

	private:
		std::shared_ptr<glasssix::excalibur::pipeline<float>> pipe;
		int device_;
		float nms_threshold;
		std::vector<float> ratio_;
		std::vector<anchor_cfg> cfg_;

		std::vector<int> feat_stride_fpn_;
		//each layer anchor shape of fpn
		std::map<std::string, std::vector<anchor_box>> anchors_fpn_;
		//each layer anchor of every points
		std::map<std::string, std::vector<anchor_box>> anchors_;
		//each layer's fpn has how many shapes of anchor = number of ratio * number of scales
		std::map<std::string, int> num_anchors_;
	};
}


#endif // _RETINA_FACE_HPP_
