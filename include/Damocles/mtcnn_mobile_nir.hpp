#ifndef _MTCNN_MOBILE_NIR_HPP_
#define _MTCNN_MOBILE_NIR_HPP_
#include "pnet_mobile_nir.hpp"
#include "rnet_mobile_nir.hpp"
#include "onet_mobile_nir.hpp"
#include "mtcnn_mobile.hpp"
#include "vdamocles.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace glasssix
{
	namespace longinus
	{
		class mtcnn_mobile_nir : public vDamocles
		{
		public:
			mtcnn_mobile_nir(int device_id, bool handle_big_face = false);
			~mtcnn_mobile_nir();

			std::vector<FaceInfomation> Detect(const unsigned char* gray, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage, int order) override;

		protected:

		private:
			bool PNet_Process(std::shared_ptr<memory::tensor<unsigned char> > &bgr_8uc3, float thresh, float nms_thresh, std::vector<float> scales, std::vector<Longinus_CNN_BBox>& pnet_result);
			bool GenerateBoundingBox(std::vector<Longinus_CNN_BBox>& bounding_bbox, std::vector<std::vector<float>>& maps, std::vector<float>& scales,
				std::vector<int>& mapH, std::vector<int>& mapW, float thresh, float nms_thresh, int stride, int cellSize, int image_width, int image_height);

			bool RNet_Process(std::vector<Longinus_CNN_BBox>& pnetBbox, std::vector<Longinus_CNN_BBox>& rnet_result, std::shared_ptr<memory::tensor<unsigned char> > &bgr_8uc3, int min_size, float thresh, float nms_thresh);
			bool ONet_Process(std::vector<Longinus_CNN_BBox>& rnetBbox, std::vector<Longinus_CNN_BBox>& onet_result, std::shared_ptr<memory::tensor<unsigned char> > & bgr_8uc3, int min_size, float thresh, float nms_thresh, bool doLandmark);
			
			pnet_mobile_nir* PNet_;
			rnet_mobile_nir* RNet_;
			onet_mobile_nir* ONet_;

			int device_id_;
			//pnet config
			const int pnet_stride = 4;
			const int pnet_size = 20;
			const int rnet_size = 24;
			const int onet_size = 64;
			bool handle_big_face_;

			std::vector<FaceBox> regressed_pading_;
		};
	}
}

#endif
