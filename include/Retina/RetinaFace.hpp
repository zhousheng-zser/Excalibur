#ifndef _RETINAFACE_HPP_
#define _RETINAFACE_HPP_

#include "Longinus/common.hpp"

#include <vector>
#include <map>

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

			std::vector<FaceInfomation> detect(const unsigned char *img_data, int min_win, int img_height, int img_width, int img_order, float threshold = 0.5);
		
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
			//shape of anchor in different fpn
			std::map<std::string, std::vector<FaceBox>> _anchors_fpn;
			//all anchors of each fpn
			std::map<std::string, std::vector<FaceBox>> _anchors;
			//number of different shapes in each fpn, ratio times scales
			std::map<std::string, int> _num_anchors;

			int device_ = -1;
			float nms_ = 0.5f;
			std::shared_ptr<glasssix::longinus::Retina_net> retina_net_;
		};
	}
}


#endif // _RETINAFACE_HPP_
