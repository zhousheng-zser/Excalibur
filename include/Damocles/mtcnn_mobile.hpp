#ifndef _MTCNN_MOBILE_HPP_
#define _MTCNN_MOBILE_HPP_
#include "pnet_mobile.hpp"
#include "rnet_mobile.hpp"
#include "onet_mobile.hpp"
#include "vdamocles.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace glasssix
{
	namespace longinus
	{
		class Longinus_CNN_OrderScore
		{
		public:
			float score;
			int oriOrder;

			Longinus_CNN_OrderScore()
			{
				memset(this, 0, sizeof(Longinus_CNN_OrderScore));
			}
		};

		class Longinus_CNN_BBox
		{
		public:
			float score;
			int row1;
			int col1;
			int row2;
			int col2;
			float area;
			bool exist;
			bool need_check_overlap_count;
			float ppoint[10];
			float regreCoord[4];
			float headpose[3];

			Longinus_CNN_BBox()
			{
				memset(this, 0, sizeof(Longinus_CNN_BBox));
			}

			~Longinus_CNN_BBox() {}
		};

		class mtcnn_mobile : public vDamocles
		{
		public:
			mtcnn_mobile(int device_id, bool handle_big_face = false);
			~mtcnn_mobile();

			std::vector<FaceInfomation> Detect(const unsigned char* gray, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage, int order);

		protected:

		private:
			bool PNet_Process(std::shared_ptr<tensor<unsigned char> > &bgr_8uc3, float thresh, float nms_thresh, std::vector<float> scales, std::vector<Longinus_CNN_BBox>& pnet_result);
			bool GenerateBoundingBox(std::vector<Longinus_CNN_BBox>& bounding_bbox, std::vector<std::vector<float>>& maps, std::vector<float>& scales,
				std::vector<int>& mapH, std::vector<int>& mapW, float thresh, float nms_thresh, int stride, int cellSize, int image_width, int image_height);

			bool RNet_Process(std::vector<Longinus_CNN_BBox>& pnetBbox, std::vector<Longinus_CNN_BBox>& rnet_result, std::shared_ptr<tensor<unsigned char> > &bgr_8uc3, int min_size, float thresh, float nms_thresh);
			bool ONet_Process(std::vector<Longinus_CNN_BBox>& rnetBbox, std::vector<Longinus_CNN_BBox>& onet_result, std::shared_ptr<tensor<unsigned char> > & bgr_8uc3, int min_size, float thresh, float nms_thresh, bool doLandmark);
			
			pnet_mobile * PNet_;
			rnet_mobile* RNet_;
			onet_mobile* ONet_;

			int device_id_;
			//omp
#ifdef _OPENMP
			const int threads_num = omp_get_num_procs();
#else
			const int threads_num = 1;
#endif
			//pnet config
			const int pnet_stride = 4;
			const int pnet_size = 20;
			const int rnet_size = 24;
			const int onet_size = 48;
			bool handle_big_face_;

			std::vector<FaceBox> regressed_pading_;
		};
	}
}

#endif
