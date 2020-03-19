#ifndef _RETINAFACE_HPP_
#define _RETINAFACE_HPP_

#include <vector>
#include <map>
#include "../../include/Longinus/common.hpp"

namespace glasssix
{
	namespace longinus
	{
		class Retina_net;

		struct anchor_win
		{
			float x_ctr;
			float y_ctr;
			float w;
			float h;
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

		class RetinaFace
		{
		public:
			RetinaFace(int device);

			~RetinaFace();

			//Test in GTX1060:
			// | model | speed | input size | preprocess time | inference | postprocess time |
			//	| :------ : | : ---- : | : -------- : | : ------------ - : | : ------ - : | : -------------- : |
			//	|  caffe | ????ms | 1920x1080 | ????ms | 61ms | ????ms      |
			//	|  caffe | ????ms | 1280ｘ720 | ????ms | 44ms | ????ms      |
			//	|  caffe | 17.3ms | 640ｘ480 | 3.9ms | 13.4ms | 1.0ms |
			std::vector<FaceInfomation> detect(const unsigned char *img_data, int img_channel, int img_height, int img_width, int img_order, float threshold = 0.5, float scales = 1.0);
		private:
			FaceBox bbox_pred(FaceBox anchor, std::vector<float> regress);
			static bool CompareBBox(const FaceInfomation &a, const FaceInfomation &b);
			std::vector<FaceInfomation> nms(std::vector<FaceInfomation> &bboxes, float threshold);

			float *cpuBuffers;

			float pixel_means[3] = { 0.0, 0.0, 0.0 };
			float pixel_stds[3] = { 1.0, 1.0, 1.0 };
			float pixel_scale = 1.0;

			int ctx_id;
			std::string network;
			float decay4;
			float nms_threshold;
			bool vote;
			bool nocrop;

			std::vector<float> _ratio;
			std::vector<anchor_cfg> cfg;

			std::vector<int> _feat_stride_fpn;
			//每一层fpn的anchor形状
			std::map<std::string, std::vector<FaceBox>> _anchors_fpn;
			//每一层所有点的anchor
			std::map<std::string, std::vector<FaceBox>> _anchors;
			//每一层fpn有几种形状的anchor
			//也就是ratio个数乘以scales个数
			std::map<std::string, int> _num_anchors;

			int device_ = -1;
			float nms_ = 0.5f;
			std::shared_ptr<glasssix::longinus::Retina_net> retina_net_;
		};
	}
}


#endif // _RETINAFACE_HPP_
