#pragma once
#ifndef _FUSTDETECTOR_HPP_
#define _FUSTDETECTOR_HPP_

#include "Classifier.hpp"
#include "Detector.hpp"
#include "FeatureMap.hpp"
#include "ModelReader.hpp"
#include <map>


namespace excalibur
{
	class FUSTDtector : public Detector
	{
		std::shared_ptr<ModelReader> CreateModelReader(ClassifierType type);
		std::shared_ptr<Classifier> CreateClassifier(ClassifierType type);
		std::shared_ptr<FeatureMap> CreateFeatureMap(ClassifierType type);

		void GetWindowData(const ImageTensor<unsigned char> & img, const Rect & wnd);

		int32_t wnd_size_;
		int32_t slide_wnd_step_x_;
		int32_t slide_wnd_step_y_;

		int32_t num_hierarchy_;
		std::vector<int32_t> hierarchy_size_;
		std::vector<int32_t> num_stage_;
		std::vector<std::vector<int32_t> > wnd_src_id_;

		std::vector<uint8_t> wnd_data_buf_;
		std::vector<uint8_t> wnd_data_;

		std::vector<std::shared_ptr<Classifier> > model_;
		std::vector<std::shared_ptr<FeatureMap> > feat_map_;
		std::map<ClassifierType, int32_t> cls2feat_idx_;
		
	public:
		FUSTDtector()
			: wnd_size_(40), slide_wnd_step_x_(4), slide_wnd_step_y_(4),
			num_hierarchy_(0) 
		{
			wnd_data_buf_.resize(wnd_size_ * wnd_size_);
			wnd_data_.resize(wnd_size_ * wnd_size_);
		}

		~FUSTDtector(){};

		virtual bool LoadModel(const std::string & model_path);
		virtual std::vector<FaceInfo> Detect(ImagePyramid* img_pyramid);

		inline virtual void SetWindowSize(int32_t size) {
			if (size >= 20)
				wnd_size_ = size;
		}

		inline virtual void SetSlideWindowStep(int32_t step_x, int32_t step_y) {
			if (step_x > 0)
				slide_wnd_step_x_ = step_x;
			if (step_y > 0)
				slide_wnd_step_y_ = step_y;
		}
	};
}

#endif // _FUSTDETECTOR_HPP_