#ifndef _READ_CAFFEMODEL_HPP_
#define _READ_CAFFEMODEL_HPP_
#include <fcntl.h>
#include <stdio.h>
#include "caffe.pb.h"
#include <fstream>  // NOLINT(readability/streams)
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <corecrt_io.h>

using namespace caffe;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::Message;

const int kProtoReadBytesLimit = std::numeric_limits<int>::max();  // Max size of 2 GB minus 1 byte.

static bool ReadProtoFromBinaryFile(const char* file, Message* net)
{
	int fd = _open(file, O_RDONLY | O_BINARY);
	if (fd == -1) return false;

	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
	bool success = net->ParseFromCodedStream(coded_input);
	delete coded_input;
	delete raw_input;
	_close(fd);
	return success;
}

static bool ReadProtoFromTextFile(const char* filename, Message* proto) {
	int fd = _open(filename, O_RDONLY);
	if (fd == -1) return false;
	FileInputStream* input = new FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, proto);
	delete input;
	_close(fd);
	return success;
}

#endif