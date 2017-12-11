#pragma once
#ifndef _MODELREADER_HPP_
#define _MODELREADER_HPP_

#include "Classifier.hpp"

namespace excalibur
{
	class ModelReader {
	public:
		ModelReader() {}
		virtual ~ModelReader() {}

		virtual bool Read(std::istream* input, Classifier* model) = 0;

		//DISABLE_COPY_AND_ASSIGN(ModelReader);
	};
}

#endif