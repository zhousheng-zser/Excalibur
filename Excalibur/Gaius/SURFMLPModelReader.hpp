#pragma once
#ifndef _SURFMLPMODELREADER_HPP_
#define _SURFMLPMODELREADER_HPP_

#include "ModelReader.hpp"

namespace excalibur
{
	class SURFMLPModelReader : public ModelReader {
	public:
		SURFMLPModelReader() {}
		virtual ~SURFMLPModelReader() {}

		virtual bool Read(std::istream* input, Classifier* model);

	private:
		std::vector<int32_t> feat_id_buf_;
		std::vector<float> weights_buf_;
		std::vector<float> bias_buf_;
	};
}

#endif // _SURFMLPMODELREADER_HPP_