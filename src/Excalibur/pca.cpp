#include "pca.hpp"

namespace glasssix
{
	namespace excalibur
	{
		pca::pca(int d, int k, int device)
		{
			initial_dimensions = d;
			final_dimensions = k;
			device_ = device;
			weights_.reset(new tensor<float>(std::vector<int>{initial_dimensions, final_dimensions}, device_));
		}


		pca::~pca()
		{

		}

		void pca::set_weights(float* weights)
		{
			weights_->set_cpu_data(weights);
		}

		void pca::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			int num = bottom->num();
			top.reset(new tensor<float>(std::vector<int>{num, final_dimensions}, device_));
			const float* bottom_data = bottom->cpu_data();
			float* top_data = top->mutable_cpu_data();
			math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, num, final_dimensions, initial_dimensions,
				1.0f, bottom_data, top_data, 0.0f, top_data);
		}


	}
}

