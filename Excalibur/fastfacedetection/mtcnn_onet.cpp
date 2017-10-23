#include "mtcnn_onet.hpp"

namespace fastface
{
	mtcnn_onet::mtcnn_onet()
	{
		NetParameter net3;
		io::readcaffemodel("..\\model\\det3-half.caffemodel", net3);
		int conv1_id = 7;
		int prelu1_id = 8;
		int conv2_id = 10;
		int prelu2_id = 11;
		int conv3_id = 13;
		int prelu3_id = 14;
		int conv4_id = 16;
		int prelu4_id = 17;
		int ip5_id = 18;
		int prelu5_id = 19;
		int ip6_1_id = 21;
		int ip6_2_id = 22;
		int ip6_3_id = 23;
		//
		conv1_para = io::readdataformcaffemodel(net3, conv1_id);
		prelu1_para = io::readdataformcaffemodel(net3, prelu1_id);
		conv2_para = io::readdataformcaffemodel(net3, conv2_id);
		prelu2_para = io::readdataformcaffemodel(net3, prelu2_id);
		conv3_para = io::readdataformcaffemodel(net3, conv3_id);
		prelu3_para = io::readdataformcaffemodel(net3, prelu3_id);
		conv4_para = io::readdataformcaffemodel(net3, conv4_id);
		prelu4_para = io::readdataformcaffemodel(net3, prelu4_id);
		ip5_para = io::readdataformcaffemodel(net3, ip5_id);
		prelu5_para = io::readdataformcaffemodel(net3, prelu5_id);
		ip6_1_para = io::readdataformcaffemodel(net3, ip6_1_id);
		ip6_2_para = io::readdataformcaffemodel(net3, ip6_2_id);
		ip6_3_para = io::readdataformcaffemodel(net3, ip6_3_id);
		//
		conv1 = new convolution(3, 16, 3, 1, 0, -1);
		conv1->set_bias_term(true);
		conv1->set_weights(conv1_para[0]);
		conv1->set_bias(conv1_para[1]);
		prelu1 = new prelu(16, false, -1);
		prelu1->setslope(prelu1_para[0]);
		pool1 = new pooling(3, 2, 0, 0, -1);
		conv2 = new convolution(16, 32, 3, 1, 0, -1);
		conv2->set_bias_term(true);
		conv2->set_weights(conv2_para[0]);
		conv2->set_bias(conv2_para[1]);
		prelu2 = new prelu(32, false, -1);
		prelu2->setslope(prelu2_para[0]);
		pool2 = new pooling(3, 2, 0, 0, -1);
		conv3 = new convolution(32, 32, 3, 1, 0, -1);
		conv3->set_bias_term(true);
		conv3->set_weights(conv3_para[0]);
		conv3->set_bias(conv3_para[1]);
		prelu3 = new prelu(32, false, -1);
		prelu3->setslope(prelu3_para[0]);
		pool3 = new pooling(2, 2, 0, 0, -1);
		conv4 = new convolution(32, 64, 2, 1, 0, -1);
		conv4->set_bias_term(true);
		conv4->set_weights(conv4_para[0]);
		conv4->set_bias(conv4_para[1]);
		prelu4 = new prelu(64, false, -1);
		prelu4->setslope(prelu4_para[0]);
		ip5 = new inner_product(std::vector<int>{1, 64, 3, 3}, 128, true, -1);
		ip5->set_weights(ip5_para[0]);
		ip5->set_bias(ip5_para[1]);
		prelu5 = new prelu(128, false, -1);
		prelu5->setslope(prelu5_para[0]);
		ip6_1 = new inner_product(std::vector<int>{1, 128, 1, 1}, 2, true, -1);
		ip6_1->set_weights(ip6_1_para[0]);
		ip6_1->set_bias(ip6_1_para[1]);
		ip6_2 = new inner_product(std::vector<int>{1, 128, 1, 1}, 4, true, -1);
		ip6_2->set_weights(ip6_2_para[0]);
		ip6_2->set_bias(ip6_2_para[1]);
		ip6_3 = new inner_product(std::vector<int>{1, 128, 1, 1}, 10, true, -1);
		ip6_3->set_weights(ip6_3_para[0]);
		ip6_3->set_bias(ip6_3_para[1]);
		prob1 = new softmax(2, -1);
	}


	mtcnn_onet::~mtcnn_onet()
	{
		delete conv1;
		delete prelu1;
		delete pool1;
		delete conv2;
		delete prelu2;
		delete pool2;
		delete conv3;
		delete prelu3;
		delete pool3;
		delete conv4;
		delete prelu4;
		delete ip5;
		delete prelu5;
		delete ip6_1;
		delete ip6_2;
		delete ip6_3;
		delete prob1;
	}

	void mtcnn_onet::Forward_cpu(const std::shared_ptr<tensor> input_data)
	{
		tensor_data.reset(new tensor(input_data->data_shape(), -1));
		float* temp = tensor_data->mutable_cpu_data();
		memcpy(temp, input_data->cpu_data(), input_data->count(0, 4) * sizeof(float));
		conv1->Forward_cpu(tensor_data, conv1_top_data);
		prelu1->Forward_cpu(conv1_top_data);
		pool1->Forward_cpu(conv1_top_data, pool1_top_data);
		conv2->Forward_cpu(pool1_top_data, conv2_top_data);
		prelu2->Forward_cpu(conv2_top_data);
		pool2->Forward_cpu(conv2_top_data, pool2_top_data);
		conv3->Forward_cpu(pool2_top_data, conv3_top_data);
		prelu3->Forward_cpu(conv3_top_data);
		pool3->Forward_cpu(conv3_top_data, pool3_top_data);
		conv4->Forward_cpu(pool3_top_data, conv4_top_data);
		prelu4->Forward_cpu(conv4_top_data);
		ip5->Forward_cpu(conv4_top_data, ip5_top_data);
		prelu5->Forward_cpu(ip5_top_data);
		ip6_1->Forward_cpu(ip5_top_data, ip6_1_top_data);
		ip6_2->Forward_cpu(ip5_top_data, ip6_2_top_data);
		ip6_3->Forward_cpu(ip5_top_data, ip6_3_top_data);
		prob1->Forward_cpu(ip6_1_top_data, prob1_top_data);
	}

	std::shared_ptr<tensor> mtcnn_onet::get_prob1()
	{
		return prob1_top_data;
	}

	std::shared_ptr<tensor> mtcnn_onet::get_ip6_2()
	{
		return ip6_2_top_data;
	}

	std::shared_ptr<tensor> mtcnn_onet::get_ip6_3()
	{
		return ip6_3_top_data;
	}

}
