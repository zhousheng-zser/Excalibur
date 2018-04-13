#pragma once
#ifndef _ALCNN_HPP_
#define _ALCNN_HPP_

#include "ipts_net.hpp"
#include "ipbbox_net.hpp"

using namespace excalibur;

namespace glasssix
{
	class alcnn
	{
		ipbbox_net* ipbbox;
		ipts_net* ipts;
		//
		int device_;
		std::shared_ptr<tensor<float>> tensor_data = nullptr;
		//
#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
#endif

	public:
		alcnn(int device);
		~alcnn();
		void Forward_IPBbox(const std::shared_ptr<tensor<float>> input_data);
		void Forward_IPTs(const std::shared_ptr<tensor<float>> input_data);

		const float* get_IPBbox_fc2_data() const
		{
			return ipbbox->get_fc2()->cpu_data();
		}

		const float* get_IPBbox_fc3_data() const
		{
			return ipbbox->get_fc3()->cpu_data();
		}

		int get_IPBbox_fc2_count() const
		{
			return ipbbox->get_fc2()->count();
		}

		int get_IPBbox_fc3_count() const
		{
			return ipbbox->get_fc3()->count();
		}

		const float* get_IPTs_fc2_data() const
		{
			return ipts->get_fc2()->cpu_data();
		}

		int get_IPTs_fc2_count() const
		{
			return ipts->get_fc2()->count();
		}
	};
}

#endif //_ALCNN_HPP_