#pragma once
#ifndef _LABBOOSTMODELREADER_HPP_
#define _LABBOOSTMODELREADER_HPP_

#include "ModelReader.hpp"
#include "LABBoostedClassifier.hpp"

namespace excalibur
{
	class LABBoostModelReader : public ModelReader {
	public:
		LABBoostModelReader() : ModelReader() {}
		virtual ~LABBoostModelReader() {}

		virtual bool Read(std::istream* input, Classifier* model);

	private:
		bool ReadFeatureParam(std::istream* input,
			LABBoostedClassifier* model);
		bool ReadBaseClassifierParam(std::istream* input,
			LABBoostedClassifier* model);

		int32_t num_bin_;
		int32_t num_base_classifer_;
	};
}

#endif