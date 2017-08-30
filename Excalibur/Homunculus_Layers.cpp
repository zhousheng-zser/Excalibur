#include "Homunculus_Layers.hpp"


namespace Excalibur
{
#ifdef CAFFEMODEL_SUPPORT

	// Serialize LayerParameter to protocol buffer
	template <typename Dtype>
	inline void Homunculus_Layers<Dtype>::ToProto(caffe::LayerParameter* param) {
		param->Clear();
		param->CopyFrom(layer_param_);
		param->clear_blobs();
		for (int i = 0; i < blobs_.size(); ++i) {
			blobs_[i]->ToProto(param->add_blobs());
		}
	}
#endif

	template <typename Dtype>
	inline void Homunculus_Layers<Dtype>::Forward(const std::vector<Pandora_Blob<Dtype>*>& bottom,
		const std::vector<Pandora_Blob<Dtype>*>& top)
	{
		Dtype loss = 0;
		Reshape(bottom, top);
		switch (exmath_->mode)
		{
		case CPU:
			Forward_cpu(bottom, top);
			break;
#ifdef USE_CUDA
		case GPU:
			Forward_gpu(bottom, top);
			break;
#endif
#ifndef x86
		case ARM:
			Forward_cpu(bottom, top);
			break;
#endif
		default:
			LOG(FATAL) << "Unknown excalibur mode.";
		}
	}

	template class Homunculus_Layers<float>;
	template class Homunculus_Layers<double>;
}