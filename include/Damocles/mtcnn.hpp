#pragma once
#ifndef _MTCNN_HPP_
#define _MTCNN_HPP_
#include "mtcnn_pnet.hpp"
#include "mtcnn_onet.hpp"
#include "mtcnn_rnet.hpp"
#include "baseCNN.hpp"
#include <omp.h>


namespace glasssix
{
	namespace longinus
	{
		class MTCNN : public BaseCNN
		{
		public:
			MTCNN(int device_id);
			~MTCNN();
			std::vector<FaceInfoX> Detect(const unsigned char* img, const int channels, const int height, const int width, const int min_size, const float* threshold, const float factor, const int stage) override;
			
		protected:
			std::vector<FaceInfoX> ProposalNet(const unsigned char* image, const int channels, const int height, const int width, int min_size, float threshold, float factor);
			std::vector<FaceInfoX> NextStage(const unsigned char* image, const int channels, const int height, const int width, std::vector<FaceInfoX> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
			void BBoxRegression(std::vector<FaceInfoX>& bboxes);
			void BBoxPadSquare(std::vector<FaceInfoX>& bboxes, int width, int height);
			void BBoxPad(std::vector<FaceInfoX>& bboxes, int width, int height);
			void GenerateBBox(std::shared_ptr<tensor<float>> confidence, std::shared_ptr<tensor<float>> reg_box, float scale, float thresh);
			std::vector<FaceInfoX> NMS(std::vector<FaceInfoX>& bboxes, float thresh, char methodType);
			float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);

		private:
			mtcnn_pnet* PNet_;
			mtcnn_rnet* RNet_;
			mtcnn_onet* ONet_;

			std::vector<FaceInfoX> candidate_boxes_;
			std::vector<FaceInfoX> total_boxes_;

			int device_id_;
			//omp
			const int threads_num = omp_get_num_procs();
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