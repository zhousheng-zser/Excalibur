#include "mtcnn_rnet.hpp"

namespace fastface
{
	mtcnn_rnet::mtcnn_rnet()
	{
		NetParameter net2;
		io::readcaffemodel("..\\model\\det2.caffemodel", net2);
		int conv1_id = 3;
		int prelu1_id = 4;
		int conv2_id = 6;
		int prelu2_id = 7;
		int conv3_id = 9;
		int prelu3_id = 10;
		int ip4_id = 11;
		int prelu4_id = 12;
		int ip5_1_id = 14;
		int ip5_2_id = 16;
		//
		conv1_para = io::readdataformcaffemodel(net2, conv1_id);
		prelu1_para = io::readdataformcaffemodel(net2, prelu1_id);
		conv2_para = io::readdataformcaffemodel(net2, conv2_id);
		prelu2_para = io::readdataformcaffemodel(net2, prelu2_id);
		conv3_para = io::readdataformcaffemodel(net2, conv3_id);
		prelu3_para = io::readdataformcaffemodel(net2, prelu3_id);
		ip4_para = io::readdataformcaffemodel(net2, ip4_id);
		prelu4_para = io::readdataformcaffemodel(net2, prelu4_id);
		ip5_1_para = io::readdataformcaffemodel(net2, ip5_1_id);
		ip5_2_para = io::readdataformcaffemodel(net2, ip5_2_id);
		//
		conv1 = new convolution(3, 28, 3, 1, 0, -1);
		conv1->set_bias_term(true);
		conv1->set_weights(conv1_para[0]);
		conv1->set_bias(conv1_para[1]);
		prelu1 = new prelu(28, false, -1);
		prelu1->setslope(prelu1_para[0]);
		pool1 = new pooling(3, 2, 0, 0, -1);
		conv2 = new convolution(28, 48, 3, 1, 0, -1);
		conv2->set_bias_term(true);
		conv2->set_weights(conv2_para[0]);
		conv2->set_bias(conv2_para[1]);
		prelu2 = new prelu(48, false, -1);
		prelu2->setslope(prelu2_para[0]);
		pool2 = new pooling(3, 2, 0, 0, -1);
		conv3 = new convolution(48, 64, 2, 1, 0, -1);
		conv3->set_bias_term(true);
		conv3->set_weights(conv3_para[0]);
		conv3->set_bias(conv3_para[1]);
		prelu3 = new prelu(64, false, -1);
		prelu3->setslope(prelu3_para[0]);
		ip4 = new inner_product(std::vector<int>{1, 64, 3, 3}, 128, true, -1);
		ip4->set_weights(ip4_para[0]);
		ip4->set_bias(ip4_para[1]);
		prelu4 = new prelu(128, false, -1);
		prelu4->setslope(prelu4_para[0]);
		ip5_1 = new inner_product(std::vector<int>{1, 128, 1, 1}, 2, true, -1);
		ip5_1->set_weights(ip5_1_para[0]);
		ip5_1->set_bias(ip5_1_para[1]);
		ip5_2 = new inner_product(std::vector<int>{1, 128, 1, 1}, 4, true, -1);
		ip5_2->set_weights(ip5_2_para[0]);
		ip5_2->set_bias(ip5_2_para[1]);
		prob1 = new softmax(2, -1);
	}


	mtcnn_rnet::~mtcnn_rnet()
	{
		delete conv1;
		delete prelu1;
		delete pool1;
		delete conv2;
		delete prelu2;
		delete pool2;
		delete conv3;
		delete prelu3;
		delete ip4;
		delete prelu4;
		delete ip5_1;
		delete ip5_2;
		delete prob1;
	}

	void mtcnn_rnet::Forward_cpu(const std::shared_ptr<tensor> input_data)
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
		ip4->Forward_cpu(conv3_top_data, ip4_top_data);
		prelu4->Forward_cpu(ip4_top_data);
		ip5_1->Forward_cpu(ip4_top_data, ip5_1_top_data);
		ip5_2->Forward_cpu(ip4_top_data, ip5_2_top_data);
		prob1->Forward_cpu(ip5_1_top_data, prob1_top_data);
	}

	std::shared_ptr<tensor> mtcnn_rnet::get_prob1()
	{
		return prob1_top_data;
	}

	std::shared_ptr<tensor> mtcnn_rnet::get_ip5_2()
	{
		return ip5_2_top_data;
	}

}
