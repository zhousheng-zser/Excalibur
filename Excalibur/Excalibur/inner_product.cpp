#include "inner_product.hpp"

namespace excalibur
{
	inner_product::inner_product(std::vector<int> input_shape_withpout_num, int num_output, bool bias_term, int device)
	{
		input_shape_without_num_ = input_shape_withpout_num;
		num_output_ = num_output;
		bias_term_ = bias_term;
		N_ = num_output_;
		device_ = device;
		//
		K_ = 1;
		for (int i = 1; i < input_shape_without_num_.size(); i++)
		{
			K_ *= input_shape_without_num_[i];
		}
		weights_ = new tensor(std::vector<int>{N_, K_}, device_);
		if (bias_term_)
		{
			bias_ = new tensor(std::vector<int>{1, N_}, device_);
		}
	}

	void inner_product::set_weights(float* weights)
	{
		weights_->set_cpu_data(weights);
	}

	void inner_product::set_bias(float* bias)
	{
		if (bias_term_)
		{
			bias_->set_cpu_data(bias);
		}
	}

	inner_product::~inner_product()
	{
		delete weights_;
		delete bias_;
	}

	void inner_product::Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		M_ = bottom->num();
		if (bias_term_)
		{
			bias_multiplier_.reset(new tensor(std::vector<int>{M_}, device_));
			math_functions::cpu_set(M_, 1.0f, bias_multiplier_->mutable_cpu_data());
		}
		top.reset(new tensor(std::vector<int>{M_, N_}, device_));
		//
		const float* bottom_data = bottom->cpu_data();
		float* top_data = (top)->mutable_cpu_data();
		const float* weight = weights_->cpu_data();
		//
		math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, M_, N_, K_, 1.0f,
			bottom_data, weight, 0.0f, top_data);
		if (bias_term_)
		{
			math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, 1.0f,
				bias_multiplier_->cpu_data(), bias_->cpu_data(), 1.0f, top_data);
		}
	}

}
