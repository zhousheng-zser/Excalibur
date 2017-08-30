#pragma once
#ifndef _GEOFRONT_NET_HPP_
#define _GEOFRONT_NET_HPP_
#include "Homunculus_Layers.hpp"
#include "io.hpp"
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
	class Geofront_Net {
	public:
		explicit Geofront_Net(const std::string& param_file);
		explicit Geofront_Net(const caffe::NetParameter& param) {
			Init(param);
		}

		~Geofront_Net();

		/// @brief Initialize a network with a NetParameter.
		void Init(const caffe::NetParameter& param);

		/**
		* @brief Run Forward and return the result.
		*
		*/
		const void Forward() {
			ForwardFromTo(0, layers_.size() - 1);
		}

		/**
		* The From and To variants of Forward and Backward operate on the
		* (topological) ordering by which the net is specified. For general DAG
		* networks, note that (1) computing from one layer to another might entail
		* extra computation on unrelated branches, and (2) computation starting in
		* the middle may be incorrect if all of the layers of a fan-in are not
		* included.
		*/
		void ForwardFromTo(int start, int end);
		void ForwardFrom(int start);
		void ForwardTo(int end);

		/**
		* @brief Reshape all layers from bottom to top.
		*
		* This is useful to propagate changes to layer sizes without running
		* a forward pass, e.g. to compute output feature size.
		*/
		void Reshape();

		// For an already initialized net, CopyTrainedLayersFrom() copies the already
		// trained layers from another net parameter instance.
		/**
		* @brief For an already initialized net, copies the pre-trained layers from
		*        another Net.
		*/
		void CopyTrainedLayersFrom(const caffe::NetParameter& param);
		void CopyTrainedLayersFrom(const std::string& trained_filename);
		/// @brief Writes the net to a proto.
		void ToProto(caffe::NetParameter* param) const;

		/// @brief returns the network name.
		const std::string& name() const { return name_; }
		/// @brief returns the layer names
		const std::vector<std::string>& layer_names() const { return layer_names_; }
		/// @brief returns the blob names
		const std::vector<std::string>& blob_names() const { return blob_names_; }
		/// @biref return the param names
		const std::vector<std::string>& param_names() const { return param_display_names_; }
		/// @brief returns the blobs
		const std::vector<std::shared_ptr<Pandora_Blob<Dtype>> >& blobs() const {
			return blobs_;
		}
		/// @brief returns the layers
		const std::vector<std::shared_ptr<Homunculus_Layers<Dtype>> >& layers() const {
			return layers_;
		}
		/// @brief all parameters
		const std::vector<std::shared_ptr<Pandora_Blob<Dtype>> >& params() const {
			return params_;
		}
		/**
		* @brief returns the bottom vecs for each layer -- usually you won't
		*        need this unless you do per-layer checks such as gradients.
		*/
		const std::vector<std::vector<Pandora_Blob<Dtype>*> >& bottom_vecs() const {
			return bottom_vecs_;
		}
		/**
		* @brief returns the top vecs for each layer -- usually you won't
		*        need this unless you do per-layer checks such as gradients.
		*/
		const std::vector<std::vector<Pandora_Blob<Dtype>*> >& top_vecs() const {
			return top_vecs_;
		}
		/**
		* @brief returns the params in every layer with id in params
		*/
		const std::vector<std::vector<int> >& param_id_vecs() const {
			return param_id_vecs_;
		}
		/// @brief returns the ids of the top blobs of layer i
		const std::vector<int> & top_ids(int i) const {
			CHECK_GE(i, 0) << "Invalid layer id";
			CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
			return top_id_vecs_[i];
		}
		/// @brief returns the ids of the bottom blobs of layer i
		const std::vector<int> & bottom_ids(int i) const {
			CHECK_GE(i, 0) << "Invalid layer id";
			CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
			return bottom_id_vecs_[i];
		}
		bool has_blob(const std::string& blob_name) const;
		const std::shared_ptr<Pandora_Blob<Dtype>> blob_by_name(const std::string& blob_name) const;
		bool has_layer(const std::string& layer_name) const;
		const std::shared_ptr<Homunculus_Layers<Dtype>> layer_by_name(const std::string& layer_name) const;

		/// @brief calculate memory usage in MB
		Dtype MemSize() const;

		/// @brief mark extra output named blob
		void MarkOutputs(const std::vector<std::string>& outs);

	protected:
		// Helpers for Init.
		/// @brief Append a new top blob to the net.
		void AppendTop(const caffe::NetParameter& param, const int layer_id,
			const int top_id, std::set<std::string>* available_blobs,
			std::map<std::string, int>* blob_name_to_idx);
		/// @brief Append a new bottom blob to the net.
		int AppendBottom(const caffe::NetParameter& param, const int layer_id,
			const int bottom_id, std::set<std::string>* available_blobs,
			std::map<std::string, int>* blob_name_to_idx);
		/// @brief Append a new parameter blob to the net.
		void AppendParam(const caffe::NetParameter& param, const int layer_id,
			const int param_id);

		/// @brief The network name
		std::string name_;
		/// @brief Individual layers in the net
		std::vector<std::shared_ptr<Homunculus_Layers<Dtype>> > layers_;
		std::vector<std::string> layer_names_;
		std::map<std::string, int> layer_names_index_;
		/// @brief the blobs storing intermediate results between the layer.
		std::vector<std::shared_ptr<Pandora_Blob<Dtype>> > blobs_;
		std::vector<std::string> blob_names_;
		std::vector<int> blob_life_time_;
		std::map<std::string, int> blob_names_index_;
		/// @brief parameters in the network.
		std::vector<std::shared_ptr<Pandora_Blob<Dtype>> > params_;
		std::vector<std::string> param_display_names_;
		std::vector<std::vector<int> > param_id_vecs_;
		std::map<std::string, int> param_names_index_;
		/// bottom_vecs stores the vectors containing the input for each layer.
		/// They don't actually host the blobs (blobs_ does), so we simply store
		/// pointers.
		std::vector<std::vector<Pandora_Blob<Dtype>*> > bottom_vecs_;
		std::vector<std::vector<int> > bottom_id_vecs_;
		/// top_vecs stores the vectors containing the output for each layer
		std::vector<std::vector<Pandora_Blob<Dtype>*> > top_vecs_;
		std::vector<std::vector<int> > top_id_vecs_;
		///
		Excalibur_MathFunctions* exmath_;
		//DISABLE_COPY_AND_ASSIGN(Net);
	};

	/// @brief Read text net parameter, like xxx.prototxt
	std::shared_ptr<caffe::NetParameter> ReadTextNetParameterFromFile(const std::string& file);
	std::shared_ptr<caffe::NetParameter> ReadTextNetParameterFromBuffer(const char* buffer, int buffer_len);
	/// @brief Read binary net parameter, like xxx.binaryproto
	std::shared_ptr<caffe::NetParameter> ReadBinaryNetParameterFromFile(const std::string& file);
	std::shared_ptr<caffe::NetParameter> ReadBinaryNetParameterFromBuffer(const char* buffer, int buffer_len);
}
#endif //_GEOFRONT_NET_HPP_