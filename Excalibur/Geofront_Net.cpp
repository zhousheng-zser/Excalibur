#include "Geofront_Net.hpp"

namespace Excalibur
{
#ifdef CAFFEMODEL_SUPPORT
	template <typename Dtype>
	Geofront_Net<Dtype>::Geofront_Net(const caffe::NetParameter& param)
	{
		Init(param);
	}

	template <typename Dtype>
	void Geofront_Net<Dtype>::Init(const caffe::NetParameter& in_param)
	{
		caffe::NetParameter filtered_param;
		FilterNet(in_param, &filtered_param);
		// Create a copy of filtered_param with splits added where necessary.
		caffe::NetParameter param;
		InsertSplits(filtered_param, &param);
		// Basically, build all the layers and set up their connections.
		name_ = param.name();
		std::map<std::string, int> blob_name_to_idx;
		std::set<std::string> available_blobs;
		memory_used_ = 0;
	}

	template <typename Dtype>
	void Geofront_Net<Dtype>::FilterNet(const caffe::NetParameter& param,
	                           caffe::NetParameter* param_filtered) {
		caffe::NetState net_state(param.state());
		param_filtered->CopyFrom(param);
		param_filtered->clear_layer();
		for (int i = 0; i < param.layer_size(); ++i) {
			const caffe::LayerParameter& layer_param = param.layer(i);
			const std::string& layer_name = layer_param.name();
			CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
				<< "Specify either include rules or exclude rules; not both.";
			// If no include rules are specified, the layer is included by default and
			// only excluded if it meets one of the exclude rules.
			bool layer_included = (layer_param.include_size() == 0);
			for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
				if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
					layer_included = false;
				}
			}
			for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
				if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
					layer_included = true;
				}
			}
			if (layer_included) {
				param_filtered->add_layer()->CopyFrom(layer_param);
			}
		}
	}

	template <typename Dtype>
	bool Geofront_Net<Dtype>::StateMeetsRule(const caffe::NetState& state,
		const caffe::NetStateRule& rule, const std::string& layer_name) {
		// Check whether the rule is broken due to phase.
		if (rule.has_phase()) 
		{
			if (rule.phase() != state.phase()) 
			{
				return false;
			}
		}
		// Check whether the rule is broken due to min level.
		if (rule.has_min_level()) 
		{
			if (state.level() < rule.min_level()) 
			{
				return false;
			}
		}
		// Check whether the rule is broken due to max level.
		if (rule.has_max_level()) 
		{
			if (state.level() > rule.max_level()) 
			{
				return false;
			}
		}
		// Check whether the rule is broken due to stage. The NetState must
		// contain ALL of the rule's stages to meet it.
		for (int i = 0; i < rule.stage_size(); ++i) {
			// Check that the NetState contains the rule's ith stage.
			bool has_stage = false;
			for (int j = 0; !has_stage && j < state.stage_size(); ++j) 
			{
				if (rule.stage(i) == state.stage(j))
				{
					has_stage = true;
				}
			}
			if (!has_stage)
			{
				return false;
			}
		}
		// Check whether the rule is broken due to not_stage. The NetState must
		// contain NONE of the rule's not_stages to meet it.
		for (int i = 0; i < rule.not_stage_size(); ++i) 
		{
			// Check that the NetState contains the rule's ith not_stage.
			bool has_stage = false;
			for (int j = 0; !has_stage && j < state.stage_size(); ++j) 
			{
				if (rule.not_stage(i) == state.stage(j))
				{
					has_stage = true;
				}
			}
			if (has_stage) 
			{
				return false;
			}
		}
		return true;
	}
#endif
	template <typename Dtype>
	Geofront_Net<Dtype>::Geofront_Net(const std::string& param_file)
	{
	}

	template <typename Dtype>
	Geofront_Net<Dtype>::~Geofront_Net()
	{
	}

	template class Geofront_Net<float>;
	template class Geofront_Net<double>;
}