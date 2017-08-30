#include "Geofront_Net.hpp"

namespace Excalibur
{
#ifdef CAFFEMODEL_SUPPORT
	
	//template <typename Dtype>
	void Geofront_Net<Dtype>::Init(const caffe::NetParameter& param)
	{
		// Basically, build all the layers and set up their connections.
		name_ = param.name();
		std::map<std::string, int> blob_name_to_idx;
		std::set<std::string> available_blobs;
		// For each layer, set up its input and output
		bottom_vecs_.resize(param.layer_size());
		top_vecs_.resize(param.layer_size());
		bottom_id_vecs_.resize(param.layer_size());
		top_id_vecs_.resize(param.layer_size());
		param_id_vecs_.resize(param.layer_size());
		for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
			// Setup layer.
			const caffe::LayerParameter& layer_param = param.layer(layer_id);
			layers_.push_back(LayerRegistry::CreateLayer(layer_param));
			layer_names_.push_back(layer_param.name());
			// Figure out this layer's input and output
			const int num_bottom = layer_param.bottom_size();
			for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
				AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
			}
			const int num_top = layer_param.top_size();
			for (int top_id = 0; top_id < num_top; ++top_id) {
				AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
			}
			// After this layer is connected, set it up.
			layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
			// Layer Parameters
			const int num_param_blobs = layers_[layer_id]->blobs().size();
			for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
				AppendParam(param, layer_id, param_id);
			}
		}
		/*CHECK_EQ(std::string(layers_[0]->type()), std::string("Input"))
			<< "Network\'s first layer should be Input Layer.";*/
		// for most case, not fully convolutional network, hold input data will be convenient
		for (int blob_id : top_id_vecs_[0]) {
			blob_life_time_[blob_id] = layers_.size();
		}
		for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
			blob_names_index_[blob_names_[blob_id]] = blob_id;
		}
		for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
			layer_names_index_[layer_names_[layer_id]] = layer_id;
		}
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