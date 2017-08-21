#pragma once
#ifndef _GEOFRONT_NET_HPP_
#define _GEOFRONT_NET_HPP_
#include "Homunculus_Layers.hpp"
#include <string>
#ifdef CAFFEMODEL_SUPPORT
#include "Atalanta_SplitsInsertor.hpp"
#endif

namespace Excalibur
{
	/**
	* @brief Connects Layer%s together into a directed acyclic graph (DAG)
	*        specified by a NetParameter.
	*
	* TODO(dox): more thorough description.
	*/
	template <typename Dtype>
	class Geofront_Net
	{
#ifdef CAFFEMODEL_SUPPORT
		/// @brief Initialize a network with a NetParameter.
		void Init(const caffe::NetParameter& param);
		// Helpers for Init.
		/**
		* @brief Remove layers that the user specified should be excluded given the current
		*        phase, level, and stage.
		*/
		static void FilterNet(const caffe::NetParameter& param,
			caffe::NetParameter* param_filtered);

		/// @brief return whether NetState state meets NetStateRule rule
		static bool StateMeetsRule(const caffe::NetState& state, const caffe::NetStateRule& rule,
			const std::string& layer_name);
#endif
	public:
#ifdef CAFFEMODEL_SUPPORT
		explicit Geofront_Net(const caffe::NetParameter& param);
#endif
		explicit Geofront_Net(const std::string& param_file);
		virtual ~Geofront_Net();

	protected:
		/// @brief The network name
		std::string name_;
		/// @brief Individual layers in the net
		std::vector<Homunculus_Layers< Dtype>* > layers_;
		std::vector<std::string> layer_names_;
		std::map<std::string, int> layer_names_index_;
		/// @brief the blobs storing intermediate results between the layer.
		std::vector<Pandora_Blob<Dtype>*> blobs_;
		std::vector<std::string> blob_names_;
		std::map<std::string, int> blob_names_index_;
		/// bottom_vecs stores the vectors containing the input for each layer.
		/// They don't actually host the blobs (blobs_ does), so we simply store
		/// pointers.
		std::vector<std::vector<Pandora_Blob<Dtype>*> > bottom_vecs_;
		std::vector<std::vector<int> > bottom_id_vecs_;
		/// top_vecs stores the vectors containing the output for each layer
		std::vector<std::vector<Pandora_Blob<Dtype>*> > top_vecs_;
		std::vector<std::vector<int> > top_id_vecs_;
		/// Vector of weight in the loss (or objective) function of each net blob,
		/// indexed by blob_id.
		std::vector<Dtype> blob_loss_weights_;
		std::vector<std::vector<int> > param_id_vecs_;
		std::vector<int> param_owners_;
		std::vector<std::string> param_display_names_;
		std::vector<std::pair<int, int> > param_layer_indices_;
		std::map<std::string, int> param_names_index_;
		/// blob indices for the input and the output of the net
		std::vector<int> net_input_blob_indices_;
		std::vector<int> net_output_blob_indices_;
		std::vector<Pandora_Blob<Dtype>*> net_input_blobs_;
		std::vector<Pandora_Blob<Dtype>*> net_output_blobs_;
		/// The parameters in the network.
		std::vector<Pandora_Blob<Dtype>* > params_;
		/// The bytes of memory used by this net
		size_t memory_used_;
		/// Whether to compute and display debug info for the net.
		bool debug_info_;
	};
}
#endif //_GEOFRONT_NET_HPP_