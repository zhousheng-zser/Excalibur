#pragma once
#ifndef _PANDORA_BLOB_HPP_
#define _PANDORA_BLOB_HPP_
#include "Avalon_Synchronizer.hpp"
#include <vector>
#ifdef CAFFEMODEL_SUPPORT
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "caffe.pb.h"
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::Message;
#endif
const int kMaxBlobAxes = 32;

namespace Excalibur
{
	template <typename Dtype>
	class Pandora_Blob
	{

	public:
		Pandora_Blob()
			:data_(), count_(0), capacity_(0), gpu_device_(-1), mode_(CPU){}
		Pandora_Blob(const std::vector<int>& shape, int gpu_device, Avalon mode);
		~Pandora_Blob();

		void Reshape(const std::vector<int>& shape);

		inline const std::vector<int>& shape() const { return shape_; }
		inline int num_axes() const { return shape_.size(); }
		inline int count() const { return count_; }

		inline int num() const { return shape()[0]; }
		inline int channels() const { return shape()[1]; }
		inline int height() const { return shape()[2]; }
		inline int width() const { return shape()[3]; }

		/**
		* @brief Compute the volume of a slice; i.e., the product of dimensions
		*        among a range of axes.
		*
		* @param start_axis The first axis to include in the slice.
		*
		* @param end_axis The first axis to exclude from the slice.
		*/
		inline int count(int start_axis, int end_axis) const 
		{
			CHECK_LE(start_axis, end_axis);
			CHECK_GE(start_axis, 0);
			CHECK_GE(end_axis, 0);
			CHECK_LE(start_axis, num_axes());
			CHECK_LE(end_axis, num_axes());
			int count = 1;
			for (int i = start_axis; i < end_axis; ++i) 
			{
				count *= shape()[i];
			}
			return count;
		}
		/**
		* @brief Compute the volume of a slice spanning from a particular first
		*        axis to the final axis.
		*
		* @param start_axis The first axis to include in the slice.
		*/
		inline int count(int start_axis) const 
		{
			return count(start_axis, num_axes());
		}

		inline int offset(const int n, const int c = 0, const int h = 0,
			const int w = 0) const 
		{
			CHECK_GE(n, 0);
			CHECK_LE(n, num());
			CHECK_GE(channels(), 0);
			CHECK_LE(c, channels());
			CHECK_GE(height(), 0);
			CHECK_LE(h, height());
			CHECK_GE(width(), 0);
			CHECK_LE(w, width());
			return ((n * channels() + c) * height() + h) * width() + w;
		}

		inline int offset(const std::vector<int>& indices) const 
		{
			CHECK_LE(indices.size(), num_axes());
			int offset = 0;
			for (int i = 0; i < num_axes(); ++i) 
			{
				offset *= shape()[i];
				if (indices.size() > i) 
				{
					CHECK_GE(indices[i], 0);
					CHECK_LT(indices[i], shape()[i]);
					offset += indices[i];
				}
			}
			return offset;
		}


		const Dtype* cpu_data() const;
		void set_cpu_data(Dtype* data);
		void set_cpu_data(Dtype* data, size_t size);
		const Dtype* gpu_data() const;
		void set_gpu_data(Dtype* data);
		Dtype* mutable_cpu_data();
		Dtype* mutable_gpu_data();

#ifdef CAFFEMODEL_SUPPORT
		void FromProto(const caffe::BlobProto& proto, bool reshape = true);
#endif
	protected:
		Avalon_Synchronizer<Dtype>* data_;
		std::vector<int> shape_;
		int count_;
		int capacity_;
		int gpu_device_;
		Avalon mode_;
	};
}
#endif //_PANDORA_BLOB_HPP_