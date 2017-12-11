#include "LABBoostModelReader.hpp"

namespace excalibur
{
	bool LABBoostModelReader::Read(std::istream* input,
		Classifier* model) {
		bool is_read;
		LABBoostedClassifier* lab_boosted_classifier =
			dynamic_cast<LABBoostedClassifier*>(model);

		input->read(reinterpret_cast<char*>(&num_base_classifer_), sizeof(int32_t));
		input->read(reinterpret_cast<char*>(&num_bin_), sizeof(int32_t));

		is_read = (!input->fail()) && num_base_classifer_ > 0 && num_bin_ > 0 &&
			ReadFeatureParam(input, lab_boosted_classifier) &&
			ReadBaseClassifierParam(input, lab_boosted_classifier);

		return is_read;
	}

	bool LABBoostModelReader::ReadFeatureParam(std::istream* input,
		LABBoostedClassifier* model) {
		int32_t x;
		int32_t y;
		for (int32_t i = 0; i < num_base_classifer_; i++) {
			input->read(reinterpret_cast<char*>(&x), sizeof(int32_t));
			input->read(reinterpret_cast<char*>(&y), sizeof(int32_t));
			model->AddFeature(x, y);
		}

		return !input->fail();
	}

	bool LABBoostModelReader::ReadBaseClassifierParam(std::istream* input,
		LABBoostedClassifier* model) {
		std::vector<float> thresh;
		thresh.resize(num_base_classifer_);
		input->read(reinterpret_cast<char*>(thresh.data()),
			sizeof(float)* num_base_classifer_);

		int32_t weight_len = sizeof(float)* (num_bin_ + 1);
		std::vector<float> weights;
		weights.resize(num_bin_ + 1);
		for (int32_t i = 0; i < num_base_classifer_; i++) {
			input->read(reinterpret_cast<char*>(weights.data()), weight_len);
			model->AddBaseClassifier(weights.data(), num_bin_, thresh[i]);
		}

		return !input->fail();
	}
}