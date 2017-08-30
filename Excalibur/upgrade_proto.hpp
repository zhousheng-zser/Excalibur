#pragma once
#ifndef _EXCALIBUR_UPGRADE_PROTO_HPP_
#define _EXCALIBUR_UPGRADE_PROTO_HPP_

#include <string>

#include "caffe.pb.h"

namespace Excalibur {

	// Return true iff the net is not the current version.
	bool NetNeedsUpgrade(const caffe::NetParameter& net_param);

	// Check for deprecations and upgrade the NetParameter as needed.
	bool UpgradeNetAsNeeded(const std::string& param_file, caffe::NetParameter* param);

	// Read parameters from a file into a NetParameter proto message.
	void ReadNetParamsFromTextFileOrDie(const std::string& param_file,
	                                    caffe::NetParameter* param);
	void ReadNetParamsFromBinaryFileOrDie(const std::string& param_file,
	                                      caffe::NetParameter* param);

	// Return true iff the Net contains any layers specified as V1LayerParameters.
	bool NetNeedsV1ToV2Upgrade(const caffe::NetParameter& net_param);

	// Perform all necessary transformations to upgrade a NetParameter with
	// deprecated V1LayerParameters.
	bool UpgradeV1Net(const caffe::NetParameter& v1_net_param, caffe::NetParameter* net_param);

	bool UpgradeV1LayerParameter(const caffe::V1LayerParameter& v1_layer_param,
	                             caffe::LayerParameter* layer_param);

	const char* UpgradeV1LayerType(const caffe::V1LayerParameter_LayerType type);

	// Return true iff the Net contains input fields.
	bool NetNeedsInputUpgrade(const caffe::NetParameter& net_param);

	// Perform all necessary transformations to upgrade input fields into layers.
	void UpgradeNetInput(caffe::NetParameter* net_param);

}  // namespace Excalibur

#endif   // _EXCALIBUR_UPGRADE_PROTO_HPP_