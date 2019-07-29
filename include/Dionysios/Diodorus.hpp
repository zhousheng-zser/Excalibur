#ifndef _DIODORUS_HPP_
#define _DIODORUS_HPP_

#include <vector>
#include <string>
#include <memory>
#include "vDiodorus.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace glasssix
{
	namespace dionysios
	{
		class Diodorus : public vDiodorus
		{
		public:
			Diodorus(int device = -1);
			virtual ~Diodorus();

			std::shared_ptr<glasssix::excalibur::tensor<unsigned char>> getFaceArea(const unsigned char* origine, int channels, int height, int width,
				std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks, int order = 1);

			bool aliveDetect(const unsigned char* srcColorVSL, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoVSL,
				int channels, int height, int width, int order,
				const unsigned char* srcColorNIR, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoNIR = std::vector<longinus::FaceRectwithFaceInfo>());

		private:
			int device_;
			unsigned char* saturate_data_;
			unsigned char* face_sobel_data_;

#ifdef USE_OPENCV
			cv::Ptr<cv::ml::SVM> svm_;
#endif
		};
	}
}

#endif// !_DIODORUS_HPP_
