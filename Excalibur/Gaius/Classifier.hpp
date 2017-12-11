#pragma once
#ifndef _CLASSIFIER_HPP_
#define _CLASSIFIER_HPP_

#include "ImageTensor.hpp"
#include "FeatureMap.hpp"

namespace excalibur
{
	enum ClassifierType {
		LAB_Boosted_Classifier,
		SURF_MLP
	};

	class Classifier
	{
	public:
		Classifier() {}
		virtual ~Classifier() {}

		virtual void SetFeatureMap(FeatureMap* feat_map) = 0;
		virtual bool Classify(float* score = nullptr, float* outputs = nullptr) = 0;

		virtual ClassifierType type() = 0;
	};
}

#endif // _CLASSIFIER_HPP_