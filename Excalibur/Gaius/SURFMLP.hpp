#pragma once
#ifndef _SURFMLP_HPP_
#define _SURFMLP_HPP_

#include "Classifier.hpp"
#include "MLP.hpp"
#include "SURFFeatureMap.hpp"

namespace excalibur
{
	class SURFMLP : public Classifier {
	public:
		SURFMLP() : Classifier(), model_(new MLP()) {}
		virtual ~SURFMLP() {}

		virtual bool Classify(float* score = nullptr, float* outputs = nullptr);

		inline virtual void SetFeatureMap(FeatureMap* feat_map) {
			feat_map_ = dynamic_cast<SURFFeatureMap*>(feat_map);
		}

		inline virtual ClassifierType type() {
			return ClassifierType::SURF_MLP;
		}

		void AddFeatureByID(int32_t feat_id);
		void AddLayer(int32_t input_dim, int32_t output_dim, const float* weights,
			const float* bias, bool is_output = false);

		inline void SetThreshold(float thresh) { thresh_ = thresh; }

	private:
		std::vector<int32_t> feat_id_;
		std::vector<float> input_buf_;
		std::vector<float> output_buf_;

		std::shared_ptr<MLP> model_;
		float thresh_;
		SURFFeatureMap* feat_map_;
	};
}

#endif