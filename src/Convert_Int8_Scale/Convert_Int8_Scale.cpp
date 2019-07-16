#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <io.h>

#include <stdio.h>
#include <limits.h>
#include <math.h>

#include <fstream>
#include <set>
#include <limits>
#include <map>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "caffe.pb.h"
#include "QuantizeLayer.h"

#include <glasssix\CaffeBinding.hpp>

#define QUANTIZE_NUM 127
#define INTERVAL_NUM 2048

using namespace std;

static bool read_proto_from_text(const char* filepath, google::protobuf::Message* message)
{
	std::ifstream fs(filepath, std::ifstream::in);
	if (!fs.is_open())
	{
		fprintf(stderr, "open failed %s\n", filepath);
		return false;
	}

	google::protobuf::io::IstreamInputStream input(&fs);
	bool success = google::protobuf::TextFormat::Parse(&input, message);

	fs.close();

	return success;
}

static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
	std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
	if (!fs.is_open())
	{
		fprintf(stderr, "open failed %s\n", filepath);
		return false;
	}

	google::protobuf::io::IstreamInputStream input(&fs);
	google::protobuf::io::CodedInputStream codedstr(&input);

	codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

	bool success = message->ParseFromCodedStream(&codedstr);

	fs.close();

	return success;
}

void listFiles(std::string dir, std::vector<std::string> &image_path)
{
	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst((dir + "/*.*").c_str(), &findData);    // 查找目录中的第一个文件
	if (handle == -1)
	{
		cout << "Failed to find first file!\n";
		return;
	}

	do
	{
		if (strcmp(findData.name, ".") != 0 && strcmp(findData.name, "..") != 0)
		{
			if (findData.attrib & _A_SUBDIR)
			{
				listFiles(dir + "/" + findData.name, image_path);
			}
			else
			{
				image_path.push_back(dir + "/" + findData.name);
			}
		}

	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);
}

void main()
{
	string proto_path = "D:/projects/caffe2ncnn/unicornModel/unicorn_model_deploy.prototxt";
	string model_path = "D:/projects/caffe2ncnn/unicornModel/unicorn_9768_usefulpart.caffemodel";
	string image_dir = "D:/projects/test_image";
	string output_path = "D:/projects/caffe2ncnn/unicornModel/my.table";
	ofstream out(output_path);

	//get all absolute image_path under image_dir
	std::vector<std::string> image_path;
	image_path.clear();	
	listFiles(image_dir, image_path);

	// load
	caffe::NetParameter proto;
	caffe::NetParameter net;

	bool s0 = read_proto_from_text(proto_path.c_str(), &proto);
	if (!s0)
	{
		fprintf(stderr, "read_proto_from_text failed\n");
		return;
	}

	bool s1 = read_proto_from_binary(model_path.c_str(), &net);
	if (!s1)
	{
		fprintf(stderr, "read_proto_from_binary failed\n");
		return;
	}

	int layer_count = proto.layer_size();

	std::vector<std::shared_ptr<QuantizeLayer>> quantize_layer_lists;

	printf("\nQuantize the kernel weight:\n");
	for (int i = 0; i < layer_count; i++)
	{
		const caffe::LayerParameter& layer = proto.layer(i);
		if (layer.type() == "Convolution" || layer.type() == "ConvolutionDepthwise" || layer.type() == "DepthwiseConvolution")
		{
			const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();

			//get kernel_size
			int kernel_size = 3;
			if (convolution_param.has_kernel_w() || convolution_param.has_kernel_h())
			{
				if (convolution_param.kernel_w() == convolution_param.kernel_h())
				{
					kernel_size = convolution_param.kernel_w();
				}
				else
				{
					std::cout << "only support kernel_w == kernel_h" << std::endl;
					return;
				}
			}
			else
			{
				kernel_size = convolution_param.kernel_size(0);
			}

			if (kernel_size == 3 || kernel_size == 1)
			{
				//get layer_name, blob_name, group, then initialize QuantizeLayer
				std::string layer_name = layer.name();
				std::string blob_name = layer.bottom(0);

				int num_group = 1;
				if (layer.type() == "ConvolutionDepthwise" || layer.type() == "DepthwiseConvolution")
				{
					num_group = convolution_param.num_output();
				}
				else
				{
					num_group = convolution_param.group();
				}

				std::shared_ptr<QuantizeLayer> quanitze_layer;
				quanitze_layer.reset(new QuantizeLayer(layer_name, blob_name, num_group));
				

				// find blob binary by layer name
				int netidx = 0;
				for (; netidx < net.layer_size(); netidx++)
				{
					if (net.layer(netidx).name() == layer.name())
					{
						break;
					}
				}

				//get weight_data, then quantize weight
				const caffe::LayerParameter& binlayer = net.layer(netidx);
				const caffe::BlobProto& weight_blob = binlayer.blobs(0);	
				quanitze_layer->quantize_weight(weight_blob.data().data(), weight_blob.data_size());

				quantize_layer_lists.push_back(quanitze_layer);
			}
		}
	}


	caffe::CaffeBinding mclc = caffe::CaffeBinding();
	int id = mclc.AddNet(proto_path, model_path, -1);
	std::vector<cv::Mat> temp;
	cv::Mat img;

	printf("\nQuantize the Activation:\n");
	for (int i = 0; i < image_path.size(); i++)
	{
		temp.clear();
		img = cv::imread(image_path[i]);
		temp.push_back(img);
		auto res = mclc.Forward(temp, id);
		for (int j = 0; j < quantize_layer_lists.size(); j++)
		{
			const float *res_data = mclc.GetBlobData(quantize_layer_lists[j]->blob_name_, id).data;
			std::vector<int> shape = mclc.GetBlobData(quantize_layer_lists[j]->blob_name_, id).size;

			int len = 1;
			for (int k = 0; k < shape.size(); k++)
			{
				len *= shape[k];
			}

			quantize_layer_lists[j]->initial_blob_max(res_data, len);
		}
	}

	for (int i = 0; i < quantize_layer_lists.size(); i++)
	{
		quantize_layer_lists[i]->initial_blob_distubution_interval();
	}
	

	printf("\nCollect histograms of activations:\n");
	for (int i = 0; i < image_path.size(); i++)
	{
		temp.clear();
		cv::Mat img = cv::imread(image_path[i]);
		temp.push_back(img);
		auto res = mclc.Forward(temp, id);
		for (int j = 0; j < quantize_layer_lists.size(); j++)
		{
			const float *res_data = mclc.GetBlobData(quantize_layer_lists[j]->blob_name_, id).data;
			std::vector<int> shape = mclc.GetBlobData(quantize_layer_lists[j]->blob_name_, id).size;
			int len = 1;
			for (int k = 0; k < shape.size(); k++)
			{
				len *= shape[k];
			}

			quantize_layer_lists[j]->initial_histograms(res_data, len);
		}
	}


	printf("\nQuantize bottom blob:\n");
	for (int i = 0; i < quantize_layer_lists.size(); i++)
	{
		quantize_layer_lists[i]->quantize_blob();
	}
	

	printf("\nSave scales:\n");
	out << "weight scales:" << std::endl;
	for (int i = 0; i < quantize_layer_lists.size(); i++)
	{
		out << "layer:" << quantize_layer_lists[i]->layer_name_ << ", group_num:" << quantize_layer_lists[i]->group_ << ", ";
		for (int g = 0; g < quantize_layer_lists[i]->group_; g++)
		{
			out << "group_" << g << "_scale:" << quantize_layer_lists[i]->weight_scale_[g] << ", ";
		}
		out << std::endl;
	}

	out << std::endl << std::endl;

	out << "bottom_blob scales:" << std::endl;
	for (int i = 0; i < quantize_layer_lists.size(); i++)
	{
		out << "layer:" << quantize_layer_lists[i]->layer_name_ << ", bottom_scale:" << quantize_layer_lists[i]->blob_scale_ << std::endl;
	}
	
	out.close();
	system("pause");
}
