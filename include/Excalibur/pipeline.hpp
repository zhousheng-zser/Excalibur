#pragma once
#ifndef _PIPELINE_HPP_
#define _PIPELINE_HPP_
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include "operation.hpp"
#include "dag.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class EXPORT_EXCALIBUR_PRIMITIVES pipeline
		{
		public:
			explicit pipeline()
			{
#ifdef USE_CUDA
				CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
				CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
#endif
				CUDA_CHECK(cudaSetDevice(device_));
#endif
			}

			explicit pipeline(std::string param_file, std::string model_file, int device = -1);

			explicit pipeline(std::string param_file, int device = -1);

			explicit pipeline(std::vector<std::string> hardcode_params, std::string model_file, int device = -1);


			~pipeline() 
			{
#ifdef USE_CUDA
				if (cublas_handle_)
				{
					CUBLAS_CHECK(cublasDestroy(cublas_handle_));
					cublas_handle_ = nullptr;
				}
#ifdef USE_CUDNN
				if (cudnn_handle_)
				{
					CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
					cudnn_handle_ = nullptr;
				}
#endif
#endif
			};

			std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>>
				forward(const std::shared_ptr<memory::tensor<Dtype>>& input_tensor)
			{
				if (device_ >= 0)
				{
#ifdef USE_CUDA
					return forward_gpu(input_tensor);
#else
					NO_GPU;
#endif
				}
				else
				{
					return forward_cpu(input_tensor);
				}
			}

			std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>>
				forward_cpu(const std::shared_ptr<memory::tensor<Dtype>>& input_tensor);

			std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>>
				forward_gpu(const std::shared_ptr<memory::tensor<Dtype>>& input_tensor);

			std::shared_ptr<memory::tensor<Dtype>> get_featmap(std::string featmap_name);

			void enable_profiler()
			{
				profile_ = true;
			}

			void disable_profiler()
			{
				profile_ = false;
			}

		private:
			void init_weights(std::string model_file);

			void init_weights();


		private:
			//param version number
			int version_;
			// pipeline name
			std::string name_ = "Unknown Pipeleine Name";
			//device
			int device_ = -1;
			//count of the operation line follows, should be exactly the count of all operation names
			int operation_count_;
			//count of all feature map, usually greater than or equals to the operation count
			int featmap_count_ = 0;
			std::vector<std::vector<std::string>> pipe_param_str_;
			// Individual operations in the pipeline
			std::vector<std::shared_ptr<operation<Dtype>>> operations_;
			std::vector<operation_param> op_params_;

			std::map<std::string, std::vector<float> > blob_scale_table;
			std::map<std::string, std::vector<float> > weight_scale_table;

			std::map<int, std::shared_ptr<memory::tensor<float>>> weights_;
			std::map<int, std::shared_ptr<memory::tensor<float>>> bias_;
			//
			//memory::pool_allocator<Dtype>* allocator;
			std::vector<std::shared_ptr<memory::tensor<Dtype>>> featmaps_;
			//std::vector<std::shared_ptr<memory::tensor<unsigned short>>> half_featmaps_;
			std::map<std::string, int> operation_names_index_;
			std::map<std::string, int> featmap_names_index_;
			std::vector<node<std::string>> op_nodes_;
			bfsvisitor<node<std::string>> bfs_ops_;
			std::vector<std::string> output_featmap_names_;
			std::vector<std::string> input_featmap_names_;

			std::vector<int> ops_execution_order_;
			std::vector<std::pair<std::vector<int>, std::vector<int>>> ops_io_featmap_;
			int find_parent_op_index(std::string inputfeatmap);
			int find_child_op_index(std::string outputfeatmap);
			std::vector<int> get_op_input_featmap_idx(std::string op_name);
			std::vector<int> get_op_output_featmap_idx(std::string op_name);
			std::vector<std::string> read_param_file(std::string filepath);

			//
			bool profile_ = false;
			int weights_mem_cost_ = 0;
			int featmap_mem_cost_ = 0;

#ifdef USE_CUDA
			cublasHandle_t cublas_handle_ = nullptr;
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle_ = nullptr;
#endif 
#endif

			DISABLE_COPY_AND_ASSIGN(pipeline);
		};
	}
}
#endif // !_PIPELINE_HPP_
