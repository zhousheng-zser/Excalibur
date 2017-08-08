#pragma once
#ifndef _ATALANTA_SPLITSINSERTOR_HPP_
#define _ATALANTA_SPLITSINSERTOR_HPP_
#include "Accelerator.hpp"
#ifdef CAFFEMODEL_SUPPORT
#include <string>
#include "caffe.pb.h"
namespace Excalibur
{
	// Copy NetParameters with SplitLayers added to replace any shared bottom
	// blobs with unique bottom blobs provided by the SplitLayer.
	void InsertSplits(const caffe::NetParameter& param, caffe::NetParameter* param_split);

	void ConfigureSplitLayer(const std::string& layer_name, const std::string& blob_name,
		const int blob_idx, const int split_count, const float loss_weight,
	                         caffe::LayerParameter* split_layer_param);

	std::string SplitLayerName(const std::string& layer_name, const std::string& blob_name,
		const int blob_idx);

	std::string SplitBlobName(const std::string& layer_name, const std::string& blob_name,
		const int blob_idx, const int split_idx);

}
#endif
#endif // _ATALANTA_SPLITSINSERTOR_HPP_