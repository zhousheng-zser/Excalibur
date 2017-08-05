#pragma once
#ifndef _HOMONCULUS_LAYERS_HPP_
#define _HOMONCULUS_LAYERS_HPP_
#include "Excalibur_MathFunctions.hpp"
#include "Pandora_Blob.hpp"
#include <string>
namespace Excalibur
{
	template <typename Dtype>
	class Homunculus_Layers
	{
	protected:
		Excalibur_MathFunctions* exmath_;
		//weights_blob_: blobs_[0];
		//bais_blob_: blobs[1];
		std::vector<Pandora_Blob<Dtype>*> blobs_;
		Avalon mode_;
		int gpu_device_;
#ifdef CAFFEMODEL_SUPPORT
		/** The protobuf that stores the layer parameters */
		caffe::LayerParameter layer_param_;
		
#endif
	public:
#ifdef CAFFEMODEL_SUPPORT
		/*explicit */Homunculus_Layers(const caffe::LayerParameter& param, int gpu_device, Avalon mode);
#endif
		Homunculus_Layers();
		virtual ~Homunculus_Layers();

		virtual void Reshape(const std::vector<Pandora_Blob<Dtype>*>& bottom,
			const std::vector<Pandora_Blob<Dtype>*>& top) = 0;
		/**
		* @brief Returns the exact number of bottom blobs required by the layer,
		*        or -1 if no exact number is required.
		*
		* This method should be overridden to return a non-negative value if your
		* layer expects some exact number of bottom blobs.
		*/
		virtual inline int ExactNumBottomBlobs() const { return -1; }

		/**
		* @brief Returns the exact number of top blobs required by the layer,
		*        or -1 if no exact number is required.
		*
		* This method should be overridden to return a non-negative value if your
		* layer expects some exact number of top blobs.
		*/
		virtual inline int ExactNumTopBlobs() const { return -1; }

		/** @brief Using the CPU device, compute the layer output. */
		virtual void Forward_cpu(const std::vector<Pandora_Blob<Dtype>*>& bottom,
			const std::vector<Pandora_Blob<Dtype>*>& top) = 0;
		/**
		* @brief Using the GPU device, compute the layer output.
		*        Fall back to Forward_cpu() if unavailable.
		*/
#ifdef USE_CUDA
		virtual void Forward_gpu(const std::vector<Pandora_Blob<Dtype>*>& bottom,
			const std::vector<Pandora_Blob<Dtype>*>& top) 
		{
			 LOG(WARNING) << "Using CPU code as backup.";
			return Forward_cpu(bottom, top);
		}
#endif
		inline Dtype Forward(const std::vector<Pandora_Blob<Dtype>*>& bottom,
			const std::vector<Pandora_Blob<Dtype>*>& top);

		// layer type name
		std::string type;
		// layer name
		std::string name;
	};
}
#endif //_HOMONCULUS_LAYERS_HPP_