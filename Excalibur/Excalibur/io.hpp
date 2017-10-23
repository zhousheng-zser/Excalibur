#pragma once
#ifndef _IO_HPP_
#define _IO_HPP_
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include "caffe.pb.h"
#include <fstream>  // NOLINT(readability/streams)
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensor.hpp"

using namespace caffe;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::Message;



namespace excalibur
{
	class io
	{
		static bool ReadProtoFromBinaryFile(const char* file, Message* net);

		static void WriteProtoToTextFile(const Message& proto, const char* filename);

	public:
		io();
		~io();

		static void images2tensor(const std::vector<cv::Mat> images, std::shared_ptr<tensor>& tensor_data, bool minus_mean = true, float scale = 0.0078125f);

		static void image2tensor(const cv::Mat image, std::shared_ptr<tensor>& tensor_data, bool minus_mean = true, float scale = 0.0078125f);

		static bool readcaffemodel(const std::string modelpath, NetParameter& net);

		static std::vector<float*> readdataformcaffemodel(NetParameter net1, int id);
	};
}
#endif // _IO_HPP_