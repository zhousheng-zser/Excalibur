#pragma once
#ifndef _RELU_LAYER_HPP_
#define _RELU_LAYER_HPP_
#include "Homunculus_Layers.hpp"

namespace Excalibur
{
	template <typename Dtype>
	class ReLU_Layer :
		public Homunculus_Layers<Dtype>
	{
	public:
#ifdef CAFFEMODEL_SUPPORT
		ReLU_Layer(const caffe::LayerParameter& param, int gpu_device, Avalon mode)
			:Homunculus_Layers<Dtype>(param, gpu_device, mode){}
#endif
		
		virtual void Reshape(const std::vector<Pandora_Blob<Dtype>*>& bottom,
			const std::vector<Pandora_Blob<Dtype>*>& top);
		
		virtual void Forward_cpu(const std::vector<Pandora_Blob<Dtype>*>& bottom,
			const std::vector<Pandora_Blob<Dtype>*>& top);

		ReLU_Layer();
		~ReLU_Layer();
		std::string test();
	};
}
#endif //_RELU_LAYER_HPP_