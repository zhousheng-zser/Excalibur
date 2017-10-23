#pragma once
#ifndef _MTCNN_ONET_HPP_
#define _MTCNN_ONET_HPP_
#include "io.hpp"
#include "support_layers.hpp"

namespace fastface
{
	class mtcnn_onet
	{
		std::vector<float*> conv1_para;
		std::vector<float*> prelu1_para;
		std::vector<float*> conv2_para;
		std::vector<float*> prelu2_para;
		std::vector<float*> conv3_para;
		std::vector<float*> prelu3_para;
		std::vector<float*> conv4_para;
		std::vector<float*> prelu4_para;
		std::vector<float*> ip5_para;
		std::vector<float*> prelu5_para;
		std::vector<float*> ip6_1_para;
		std::vector<float*> ip6_2_para;
		std::vector<float*> ip6_3_para;
		//
		convolution* conv1;
		prelu *prelu1;
		pooling *pool1;
		convolution* conv2;
		prelu *prelu2;
		pooling *pool2;
		convolution* conv3;
		prelu *prelu3;
		pooling *pool3;
		convolution* conv4;
		prelu *prelu4;
		inner_product* ip5;
		prelu *prelu5;
		inner_product* ip6_1;
		inner_product* ip6_2;
		inner_product* ip6_3;
		softmax *prob1;
		//
		std::shared_ptr<tensor> tensor_data = nullptr;
		std::shared_ptr<tensor> conv1_top_data = nullptr;
		std::shared_ptr<tensor> pool1_top_data = nullptr;
		std::shared_ptr<tensor> conv2_top_data = nullptr;
		std::shared_ptr<tensor> pool2_top_data = nullptr;
		std::shared_ptr<tensor> conv3_top_data = nullptr;
		std::shared_ptr<tensor> pool3_top_data = nullptr;
		std::shared_ptr<tensor> conv4_top_data = nullptr;
		std::shared_ptr<tensor> ip5_top_data = nullptr;
		std::shared_ptr<tensor> ip6_1_top_data = nullptr;
		std::shared_ptr<tensor> ip6_2_top_data = nullptr;
		std::shared_ptr<tensor> ip6_3_top_data = nullptr;
		std::shared_ptr<tensor> prob1_top_data = nullptr;
	public:
		mtcnn_onet();
		~mtcnn_onet();
		void Forward_cpu(const std::shared_ptr<tensor> input_data);
		std::shared_ptr<tensor> get_prob1();
		std::shared_ptr<tensor> get_ip6_2();
		std::shared_ptr<tensor> get_ip6_3();
	};
}

#endif