#pragma once
#ifndef _MTCNN_RNET_HPP_
#define _MTCNN_RNET_HPP_
#include "../Excalibur/io.hpp"
#include "../Excalibur/support_layers.hpp"

using namespace excalibur;

namespace fastface
{
	class mtcnn_rnet
	{
		std::vector<float*> conv1_para;
		std::vector<float*> prelu1_para;
		std::vector<float*> conv2_para;
		std::vector<float*> prelu2_para;
		std::vector<float*> conv3_para;
		std::vector<float*> prelu3_para;
		std::vector<float*> ip4_para;
		std::vector<float*> prelu4_para;
		std::vector<float*> ip5_1_para;
		std::vector<float*> ip5_2_para;
		//
		convolution* conv1;
		prelu *prelu1;
		pooling *pool1;
		convolution* conv2;
		prelu *prelu2;
		pooling *pool2;
		convolution* conv3;
		prelu *prelu3;
		inner_product* ip4;
		prelu *prelu4;
		inner_product* ip5_1;
		inner_product* ip5_2;
		softmax *prob1;
		//
		std::shared_ptr<tensor> tensor_data = nullptr;
		std::shared_ptr<tensor> conv1_top_data = nullptr;
		std::shared_ptr<tensor> pool1_top_data = nullptr;
		std::shared_ptr<tensor> conv2_top_data = nullptr;
		std::shared_ptr<tensor> pool2_top_data = nullptr;
		std::shared_ptr<tensor> conv3_top_data = nullptr;
		std::shared_ptr<tensor> ip4_top_data = nullptr;
		std::shared_ptr<tensor> ip5_1_top_data = nullptr;
		std::shared_ptr<tensor> ip5_2_top_data = nullptr;
		std::shared_ptr<tensor> prob1_top_data = nullptr;
		//
	public:
		mtcnn_rnet();
		~mtcnn_rnet();
		void Forward_cpu(const std::shared_ptr<tensor> input_data);
		std::shared_ptr<tensor> get_prob1();
		std::shared_ptr<tensor> get_ip5_2();
		/*std::shared_ptr<tensor> get_data()
		{
			return ip4_top_data;
		}*/
	};
}

#endif