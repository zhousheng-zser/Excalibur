#pragma once
#ifndef _MTCNN_HPP_
#define _MTCNN_HPP_
#include "mtcnn_pnet.hpp"
#include "mtcnn_onet.hpp"
#include "mtcnn_rnet.hpp"
#include "vdamocles.hpp"
#include <omp.h>


namespace glasssix
{
	namespace longinus
	{
		class MTCNN : public vDamocles
		{
		public:
			MTCNN(int device_id);
			~MTCNN();
			std::vector<FaceInfomation> Detect(const unsigned char* img, const int channels, const int height, const int width, 
				const int min_size, const float* threshold, const float factor, const int stage, int order) override;
			
		protected:
			std::vector<FaceInfomation> ProposalNet(const std::shared_ptr<tensor<float>> &image, int min_size, float threshold, float factor, orderType order = orderType::NHWC);
			std::vector<FaceInfomation> NextStage(const std::shared_ptr<tensor<float>> &image, std::vector<FaceInfomation> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold, orderType order = orderType::NHWC);
			void GenerateBBox(const std::shared_ptr<tensor<float>> &confidence, const std::shared_ptr<tensor<float>> &reg_box, float scale, float thresh);
			std::vector<FaceInfomation> NMS(std::vector<FaceInfomation>& bboxes, float thresh, char methodType);
			void refine(std::vector<FaceInfomation> &vecFaceInfomation, const int &height, const int &width, bool square);

		private:
			mtcnn_pnet* PNet_;
			mtcnn_rnet* RNet_;
			mtcnn_onet* ONet_;

			std::vector<FaceInfomation> candidate_boxes_;
			std::vector<FaceInfomation> total_boxes_;

			int device_id_;
			//pnet config
			const float pnet_stride = 2;
			const float pnet_cell_size = 12;
			const int pnet_max_detect_num = 5000;
			//mean & std
			const float mean_val = 127.5f;
			const float std_val = 0.0078125f;
			//minibatch size
			const int step_size = 128;
		};
	}
}

#endif
