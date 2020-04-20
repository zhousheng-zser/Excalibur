#include <string>
#include <fstream>  // NOLINT(readability/streams)
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>

#include <fcntl.h>
#include <corecrt_io.h>

#include "Merge_Conv_BN_Scale/caffe.pb.h"

#include <glasssix/logger.hpp>

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

static void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
	std::fstream output(filename, std::ios::out | std::ios::trunc | std::ios::binary);
	CHECK(proto.SerializeToOstream(&output));
}

static void WriteProtoToTextFile(const Message& proto, const char* filename) {
	int fd = _open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	FileOutputStream* output = new FileOutputStream(fd);
	CHECK(google::protobuf::TextFormat::Print(proto, output));
	delete output;
	_close(fd);
}

void Merge_Conv_BN_Scale(std::string input_weights, std::string output_prefix)
{
	NetParameter param;
	NetParameter param_merged;
	::ReadProtoFromBinaryFile(input_weights.c_str(), &param);

	param_merged.CopyFrom(param);
	param_merged.clear_layer();

	for (size_t n = 0; n < param.layer_size();)
	{
		std::string layer_type = param.layer(n).type();
		
		if ((layer_type == "Convolution" || layer_type == "DepthwiseConvolution" || layer_type == "ConvolutionDepthwise") && (param.layer_size() > (n + 2)))
		{
			if ((param.layer(n + 1).type() == "BatchNorm") && (param.layer(n+2).type() == "Scale"))
			{
				std::vector<std::string> conv_layer_top_names;
				std::vector<std::string> bn_layer_top_names;
				std::vector<std::string> bn_layer_bottom_names;
				std::vector<std::string> scale_layer_bottom_names;
				std::vector<std::string> scale_layer_top_names;

				int conv_layer_index = n;
				int bn_layer_index = n+1;
				int scale_layer_index = n+2;

				LayerParameter conv_layer_param = param.layer(conv_layer_index);
				LayerParameter bn_layer_param = param.layer(bn_layer_index);
				LayerParameter scale_layer_param = param.layer(scale_layer_index);

				for (int j = 0; j < conv_layer_param.top_size(); j++)
					conv_layer_top_names.push_back(conv_layer_param.top(j));

				for (int j = 0; j < bn_layer_param.top_size(); j++)
					bn_layer_top_names.push_back(bn_layer_param.top(j));
				for (int j = 0; j < bn_layer_param.bottom_size(); j++)
					bn_layer_bottom_names.push_back(bn_layer_param.bottom(j));

				for (size_t j = 0; j < scale_layer_param.bottom_size(); j++)
					scale_layer_bottom_names.push_back(scale_layer_param.bottom(j));

				for (size_t j = 0; j < scale_layer_param.top_size(); j++)
					scale_layer_top_names.push_back(scale_layer_param.top(j));

				CHECK((conv_layer_top_names[0] == bn_layer_bottom_names[0]) && (bn_layer_top_names[0] == scale_layer_bottom_names[0]));

				int N = conv_layer_param.convolution_param().num_output();
				int kernel_h = conv_layer_param.convolution_param().kernel_h();
				int kernel_w = conv_layer_param.convolution_param().kernel_w();
				if (kernel_h == 0 || kernel_w == 0)
				{
					kernel_h = kernel_w = conv_layer_param.convolution_param().kernel_size(0);
				}
				CHECK(kernel_h && kernel_w);

				int C = conv_layer_param.blobs(0).data_size() / N / kernel_h / kernel_w;
				int spatial_dim = kernel_h * kernel_w;
				const float bn_scale_factor = bn_layer_param.mutable_blobs(2)->data(0) == 0 ?
					0 : 1 / bn_layer_param.mutable_blobs(2)->data(0);

				BlobProto* blob_w = conv_layer_param.mutable_blobs(0);

				BlobProto* blob_b = nullptr;
				if (conv_layer_param.blobs_size() > 1)
				{
					blob_b = conv_layer_param.mutable_blobs(1);
				}
				else
				{
					blob_b = conv_layer_param.add_blobs();
					::caffe::BlobShape* bias_shape = blob_b->mutable_shape();					
					bias_shape->add_dim(N);

					for (int i = 0; i < N; i++)
					{
						blob_b->add_data(0.0);
					}
					
					conv_layer_param.mutable_convolution_param()->set_bias_term(true);
				}

				for (int i = 0; i < N; i++)
				{
					float mean = bn_layer_param.mutable_blobs(0)->data(i) * bn_scale_factor;
					float variance = pow(bn_layer_param.mutable_blobs(1)->data(i) * bn_scale_factor + bn_layer_param.batch_norm_param().eps(), 0.5);
					float gama = scale_layer_param.mutable_blobs(0)->data(i);

					for (size_t j = 0; j < C; j++)
					{
						for (size_t k = 0; k < spatial_dim; k++)
						{
							float w_old = blob_w->data(i * C * spatial_dim + j * spatial_dim + k);

							float w_new = w_old * gama / variance;

							blob_w->set_data(i * C * spatial_dim + j * spatial_dim + k, w_new);
						}
					}

					float beta = 0.0;
					if (scale_layer_param.scale_param().bias_term())
					{
						beta = scale_layer_param.mutable_blobs(1)->data(i);
					}

					float b_old = blob_b->data(i);

					float b_new = (b_old - mean) * gama / variance + beta;
					blob_b->set_data(i, b_new);
				}

				if (param.layer_size() > (scale_layer_index + 1))
				{
					LayerParameter *behind_scale_layer_param = param.mutable_layer(scale_layer_index + 1);
					std::vector<std::string> behind_scale_layer_bottom_names;
					for (size_t i = 0; i < behind_scale_layer_param->bottom_size(); i++)
					{
						behind_scale_layer_bottom_names.push_back(behind_scale_layer_param->bottom(i));
					}
					for (size_t i = 0; i < behind_scale_layer_bottom_names.size(); i++)
					{
						if (scale_layer_top_names[0] == behind_scale_layer_bottom_names[i])
						{
							behind_scale_layer_param->set_bottom(i, conv_layer_top_names[0]);
						}
					}
				}
				else
				{
					conv_layer_param.set_top(0, scale_layer_top_names[0]);
				}

				param_merged.add_layer()->CopyFrom(conv_layer_param);

				n += 3;
			}
			else
			{
				param_merged.add_layer()->CopyFrom(param.layer(n));
				n++;
			}
		}
		else
		{
			param_merged.add_layer()->CopyFrom(param.layer(n));
			n++;
		}
	}

	::WriteProtoToBinaryFile(param_merged, (output_prefix + ".caffemodel").c_str());

	for (size_t i = 0; i < param_merged.layer_size(); i++)
	{
		param_merged.mutable_layer(i)->clear_blobs();
	}
	::WriteProtoToTextFile(param_merged, (output_prefix + ".prototxt").c_str());
}

void Usage(char *argv)
{
	std::cout << "Usage: " << std::endl
		<< argv << " " << "[src_caffemodel]"
		<< " " << "[dst_caffemode_prefix]" << std::endl;
}

int main(int argc, char *argv[])
{
	std::string src_caffemodel = "D:/projects/data/unicornModel/gaius/new/mobile_unicorn_8805_usefulpart.caffemodel";
	std::string dst_caffemode_prefix = "D:/projects/data/unicornModel/gaius/new/mobile_unicorn_8805_usefulpart_merged";

	Merge_Conv_BN_Scale(src_caffemodel, dst_caffemode_prefix);

	system("pause");
	return 0;
}