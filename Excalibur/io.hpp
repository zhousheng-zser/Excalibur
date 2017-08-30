#pragma once
#ifndef _IO_HPP_
#define _IO_HPP_

#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "Accelerator.hpp"
#include "caffe.pb.h"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace Excalibur {

	using ::google::protobuf::Message;

	bool ReadProtoFromTextFile(const char* filename, Message* proto);

	inline bool ReadProtoFromTextFile(const std::string& filename, Message* proto) {
		return ReadProtoFromTextFile(filename.c_str(), proto);
	}

	inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
		CHECK(ReadProtoFromTextFile(filename, proto));
	}

	inline void ReadProtoFromTextFileOrDie(const std::string& filename, Message* proto) {
		ReadProtoFromTextFileOrDie(filename.c_str(), proto);
	}

	void WriteProtoToTextFile(const Message& proto, const char* filename);
	inline void WriteProtoToTextFile(const Message& proto, const std::string& filename) {
		WriteProtoToTextFile(proto, filename.c_str());
	}

	bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

	inline bool ReadProtoFromBinaryFile(const std::string& filename, Message* proto) {
		return ReadProtoFromBinaryFile(filename.c_str(), proto);
	}

	inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
		CHECK(ReadProtoFromBinaryFile(filename, proto));
	}

	inline void ReadProtoFromBinaryFileOrDie(const std::string& filename,
		Message* proto) {
		ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
	}

	void WriteProtoToBinaryFile(const Message& proto, const char* filename);
	inline void WriteProtoToBinaryFile(
		const Message& proto, const std::string& filename) {
		WriteProtoToBinaryFile(proto, filename.c_str());
	}

}  // namespace Excalibur

#endif //_IO_HPP_

