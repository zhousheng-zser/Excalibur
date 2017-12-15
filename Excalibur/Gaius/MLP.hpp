#pragma once
#ifndef _MLP_HPP_
#define _MLP_HPP_
#include <vector>
#include <memory>
#include "MLPLayer.hpp"

namespace excalibur
{
	class MLP {
	public:
		MLP() {}
		~MLP() {}

		void Compute(const float* input, float* output);

		inline int32_t GetInputDim() const {
			return layers_[0]->GetInputDim();
		}

		inline int32_t GetOutputDim() const {
			return layers_.back()->GetOutputDim();
		}

		inline int32_t GetLayerNum() const {
			return static_cast<int32_t>(layers_.size());
		}

		void AddLayer(int32_t inputDim, int32_t outputDim, const float* weights,
			const float* bias, bool is_output = false);

	private:
		std::vector<std::shared_ptr<MLPLayer> > layers_;
		std::vector<float> layer_buf_[2];
	};
}

#endif