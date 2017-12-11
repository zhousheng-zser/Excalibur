#pragma once
#ifndef _LABBOOSTEDCLASSIFIER_HPP_
#define _LABBOOSTEDCLASSIFIER_HPP_

#include "LABBaseClassifier.hpp"
#include "LABFeatureMap.hpp"

namespace excalibur
{
	class LABBoostedClassifier : public Classifier {
	public:
		LABBoostedClassifier() : use_std_dev_(true) {}
		virtual ~LABBoostedClassifier() {}

		virtual bool Classify(float* score = nullptr, float* outputs = nullptr);

		inline virtual ClassifierType type() {
			return ClassifierType::LAB_Boosted_Classifier;
		}

		void AddFeature(int32_t x, int32_t y);
		void AddBaseClassifier(const float* weights, int32_t num_bin, float thresh);

		inline virtual void SetFeatureMap(FeatureMap* featMap) {
			feat_map_ = dynamic_cast<LABFeatureMap*>(featMap);
		}

		inline void SetUseStdDev(bool useStdDev) { use_std_dev_ = useStdDev; }

	private:
		static const int32_t kFeatGroupSize = 10;
		const float kStdDevThresh = 10.0f;

		std::vector<LABFeature> feat_;
		std::vector<std::shared_ptr<LABBaseClassifier> > base_classifiers_;
		LABFeatureMap* feat_map_;
		bool use_std_dev_;
	};
}

#endif