#include "Homunculus_Layers.hpp"


namespace Excalibur
{
#ifdef CAFFEMODEL_SUPPORT
	template <typename Dtype>
	Homunculus_Layers<Dtype>::Homunculus_Layers(const caffe::LayerParameter& param)
	{
		/*gpu_device_ = gpu_device;
		mode_ = mode;
		exmath_ = new Excalibur_MathFunctions(gpu_device_, mode_);*/
		layer_param_ = param;
		if (layer_param_.blobs_size() > 0)
		{
			blobs_.resize(layer_param_.blobs_size());
			for (int i = 0; i < layer_param_.blobs_size(); ++i)
			{
				int dim_size = layer_param_.blobs(i).shape().dim_size();
				if (dim_size==1)
				{
					blobs_[i] = new Pandora_Blob<Dtype>(
						std::vector<int>{(int)layer_param_.blobs(i).shape().dim(0),
						1, 1, 1
						},
						gpu_device_, mode_);
					blobs_[i]->FromProto(layer_param_.blobs(i));
				}
				if (dim_size==4)
				{
					blobs_[i] = new Pandora_Blob<Dtype>(
						std::vector<int>{(int)layer_param_.blobs(i).shape().dim(0),
						(int)layer_param_.blobs(i).shape().dim(1),
						(int)layer_param_.blobs(i).shape().dim(2),
						(int)layer_param_.blobs(i).shape().dim(3)},
						gpu_device_, mode_);
					blobs_[i]->FromProto(layer_param_.blobs(i));
				}
				
				
			}
		}
		type = layer_param_.type();
		name = layer_param_.name();
	}

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
	Homunculus_Layers<Dtype>::Homunculus_Layers()
	{
		gpu_device_ = -1;
		mode_ = CPU;
		exmath_ = new Excalibur_MathFunctions(gpu_device_, mode_);
	}

	template <typename Dtype>
	Homunculus_Layers<Dtype>::~Homunculus_Layers()
	{
		for (int i = 0; i < blobs_.size(); i++)
		{
			delete blobs_[i];
		}
		blobs_.clear();
		delete exmath_;
	}

	template <typename Dtype>
	void Homunculus_Layers<Dtype>::SetDevice(Avalon mode, int device)
	{
		gpu_device_ = device;
		mode_ = mode;
		exmath_ = new Excalibur_MathFunctions(gpu_device_, mode_);
	}

	template <typename Dtype>
	inline Dtype Homunculus_Layers<Dtype>::Forward(const std::vector<Pandora_Blob<Dtype>*>& bottom,
		const std::vector<Pandora_Blob<Dtype>*>& top)
	{
		Dtype loss = 0;
		Reshape(bottom, top);
		switch (mode_)
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
		return loss;
	}

	template class Homunculus_Layers<float>;
	template class Homunculus_Layers<double>;
}