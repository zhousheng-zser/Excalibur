#pragma once
#ifndef _LABBASECLASSIFIER_HPP_
#define _LABBASECLASSIFIER_HPP_

#include "Classifier.hpp"
#include "LABFeatureMap.hpp"

namespace excalibur
{
	class LABBaseClassifier {
	public:
		LABBaseClassifier()
			: num_bin_(255), thresh_(0.0f) {
			weights_.resize(num_bin_ + 1);
		}

		~LABBaseClassifier() {}

		void SetWeights(const float* weights, int32_t num_bin)
		{
			weights_.resize(num_bin + 1);
			num_bin_ = num_bin;
			std::copy(weights, weights + num_bin_ + 1, weights_.begin());
		}

		void SetThreshold(float thresh) { thresh_ = thresh; }

		int32_t num_bin() const { return num_bin_; }
		float weights(int32_t val) const { return weights_[val]; }
		float threshold() const { return thresh_; }

	private:
		int32_t num_bin_;

		std::vector<float> weights_;
		float thresh_;
	};
}

#endif