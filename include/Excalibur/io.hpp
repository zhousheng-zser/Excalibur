#pragma once
#ifndef _IO_HPP_
#define _IO_HPP_
#include <glasssix/tensor.hpp>
//#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#ifdef CAFFEMODEL_SUPPORT
#include "caffe.pb.h"
#include <fstream>  // NOLINT(readability/streams)
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#endif
#include <iostream>
#include <string>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#ifdef CAFFEMODEL_SUPPORT
using namespace caffe;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::Message;
#endif

namespace glasssix
{
	namespace excalibur
	{
		class io
		{

#ifdef CAFFEMODEL_SOPPORT
			static bool ReadProtoFromBinaryFile(const char* file, Message* net);

			static void WriteProtoToTextFile(const Message& proto, const char* filename);
#endif
		public:
			io();
			~io();
			static void bytes2tensor(const unsigned char* bytes, int num, int channel, int height, int width,
				std::shared_ptr<tensor<float>>& tensor_data, bool minus_mean = true, float scale = 0.0078125f);
			static void bytes2tensor(const char* bytes, int num, int channel, int height, int width,
				std::shared_ptr<tensor<float>>& tensor_data, bool minus_mean = true, float scale = 0.0078125f);
#ifdef USE_OPENCV
			static void images2tensor(const std::vector<cv::Mat> images, std::shared_ptr<tensor<float>>& tensor_data, bool minus_mean = true, float scale = 0.0078125f);

			static void image2tensor(const cv::Mat image, std::shared_ptr<tensor<float>>& tensor_data, bool minus_mean = true, float scale = 0.0078125f);
#endif
#ifdef CAFFEMODEL_SOPPORT
			static bool readcaffemodel(const std::string modelpath, NetParameter& net);

			static std::vector<float*> readdataformcaffemodel(NetParameter net1, int id);
#endif
		};
	}
}

#endif // _IO_HPP_
