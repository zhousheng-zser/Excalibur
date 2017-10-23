#include "mtcnn_pnet.hpp"

namespace fastface
{
	mtcnn_pnet::mtcnn_pnet()
	{
		NetParameter net1;
		io::readcaffemodel("..\\model\\det1.caffemodel", net1);
		int conv1_id = 4;
		int prelu1_id = 5;
		int conv2_id = 7;
		int prelu2_id = 8;
		int conv3_id = 9;
		int prelu3_id = 10;
		int conv4_1_id = 12;
		int conv4_2_id = 14;

		device_ = -1;
		if (device_>=0)
		{
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
		}
		//
		/*std::string path = "..\\model\\pnet_data.hpp";
		std::ofstream out(path, std::ios::app);
		out << "#ifndef _PNET_DATA_HPP_" << std::endl;
		out << "#define _PNET_DATA_HPP_" << std::endl;
		out << std::endl << std::endl;
		out << "static const float model_weights_PNet_[] = {" << std::endl;*/
		conv1_para = io::readdataformcaffemodel(net1, conv1_id);
		prelu1_para = io::readdataformcaffemodel(net1, prelu1_id);
		conv2_para = io::readdataformcaffemodel(net1, conv2_id);
		prelu2_para = io::readdataformcaffemodel(net1, prelu2_id);
		conv3_para = io::readdataformcaffemodel(net1, conv3_id);
		prelu3_para = io::readdataformcaffemodel(net1, prelu3_id);
		conv4_1_para = io::readdataformcaffemodel(net1, conv4_1_id);
		conv4_2_para = io::readdataformcaffemodel(net1, conv4_2_id);

		//out << "#endif // _PNET_DATA_HPP_" << std::endl;
		//
		conv1 = new convolution(3, 10, 3, 1, 0, device_);
		conv1->set_bias_term(true);
		conv1->set_weights(conv1_para[0]);
		conv1->set_bias(conv1_para[1]);
		prelu1 = new prelu(10, false, device_);
		prelu1->setslope(prelu1_para[0]);
		pool1 = new pooling(2, 2, 0, 0, device_);
		conv2 = new convolution(10, 16, 3, 1, 0, device_);
		conv2->set_bias_term(true);
		conv2->set_weights(conv2_para[0]);
		conv2->set_bias(conv2_para[1]);
		prelu2 = new prelu(16,false, device_);
		prelu2->setslope(prelu2_para[0]);
		conv3 = new convolution(16, 32, 3, 1, 0, device_);
		conv3->set_bias_term(true);
		conv3->set_weights(conv3_para[0]);
		conv3->set_bias(conv3_para[1]);
		prelu3 = new prelu(32, false, device_);
		prelu3->setslope(prelu3_para[0]);
		conv4_1 = new convolution(32, 2, 1, 1, 0, device_);
		conv4_1->set_bias_term(true);
		conv4_1->set_weights(conv4_1_para[0]);
		conv4_1->set_bias(conv4_1_para[1]);
		conv4_2 = new convolution(32, 4, 1, 1, 0, device_);
		conv4_2->set_bias_term(true);
		conv4_2->set_weights(conv4_2_para[0]);
		conv4_2->set_bias(conv4_2_para[1]);
		prob1 = new softmax(2, device_);
		//
	}


	mtcnn_pnet::~mtcnn_pnet()
	{
		delete conv1;
		delete prelu1;
		delete pool1;
		delete conv2;
		delete prelu2;
		delete conv3;
		delete prelu3;
		delete conv4_1;
		delete conv4_2;
		delete prob1;
	}

	void mtcnn_pnet::Forward_cpu(const std::shared_ptr<tensor> input_data)
	{
		tensor_data.reset(new tensor(input_data->data_shape(), -1));
		float* temp = tensor_data->mutable_cpu_data();
		memcpy(temp, input_data->cpu_data(), input_data->count(0, 4) * sizeof(float));
		conv1->Forward_cpu(tensor_data, conv1_top_data);
		prelu1->Forward_cpu(conv1_top_data);
		pool1->Forward_cpu(conv1_top_data, pool1_top_data);
		conv2->Forward_cpu(pool1_top_data, conv2_top_data);
		prelu2->Forward_cpu(conv2_top_data);
		conv3->Forward_cpu(conv2_top_data, conv3_top_data);
		prelu3->Forward_cpu(conv3_top_data);
		conv4_1->Forward_cpu(conv3_top_data, conv4_1_top_data);
		conv4_2->Forward_cpu(conv3_top_data, conv4_2_top_data);
		prob1->Forward_cpu(conv4_1_top_data, prob1_top_data);
	}

#ifdef USE_CUDA
	void mtcnn_pnet::Forward_native_gpu(const std::shared_ptr<tensor> input_data)
	{
		tensor_data.reset(new tensor(input_data->data_shape(), device_));
		float* temp = tensor_data->mutable_gpu_data();
		math_functions::excalibur_copy(input_data->count(0, 4), input_data->gpu_data(), temp, device_);
		conv1->Forward_native_gpu(cublas_handle_, tensor_data, conv1_top_data);
		prelu1->Forward_native_gpu(conv1_top_data);
		pool1->Forward_native_gpu(conv1_top_data, pool1_top_data);
		conv2->Forward_native_gpu(cublas_handle_, pool1_top_data, conv2_top_data);
		prelu2->Forward_native_gpu(conv2_top_data);
		conv3->Forward_native_gpu(cublas_handle_, conv2_top_data, conv3_top_data);
		prelu3->Forward_native_gpu(conv3_top_data);
		conv4_1->Forward_native_gpu(cublas_handle_, conv3_top_data, conv4_1_top_data);
		conv4_2->Forward_native_gpu(cublas_handle_, conv3_top_data, conv4_2_top_data);
	}

#endif

	std::shared_ptr<tensor> mtcnn_pnet::get_prob1()
	{
		return prob1_top_data;
	}

	std::shared_ptr<tensor> mtcnn_pnet::get_conv4_2()
	{
		return conv4_2_top_data;
	}

}
