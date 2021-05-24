#include "dag.hpp"
#include "operation.hpp"
#include "Excalibur/pipeline.hpp"
#include "Primitives/profiler.hpp"
#include "Excalibur/operation_reflector.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include <algorithm>

namespace glasssix
{
    namespace excalibur
    {
        template <typename Dtype>
        class pipeline<Dtype>::impl
        {
        public:
            explicit impl()
            {
#ifdef USE_CUDA
                CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
                CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
#endif
                CUDA_CHECK(cudaSetDevice(device_));
#endif
            }

            explicit impl(std::string_view param_file, std::string_view model_file, int device) : impl{read_param_file(param_file), model_file, device}
            {
            }

            explicit impl(std::string_view param_file, int device) : impl{read_param_file(param_file), std::string(), device}
            {
            }

            explicit impl(const std::vector<std::string> &hardcode_params, std::string_view model_file, int device = -1)
            {
#ifdef USE_CUDA
                CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
                CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
#endif
                if (device >= 0)
                {
                    CUDA_CHECK(cudaSetDevice(device));
                    device_ = device;
                }
#endif
                decltype(auto) lines = hardcode_params;
                if (lines.size() <= 0)
                {
                    LOG(FATAL) << "Incorrect param file.";
                }
                std::string param_version = split_string(lines[0], " ")[0];
                if (param_version != "glsv1" && param_version != "7767517")
                {
                    LOG(FATAL) << "Incorrect param file version.";
                }
                if (split_string(lines[0], " ").size() > 1)
                {
                    name_ = split_string(lines[0], " ")[1];
                }
                for (size_t i = 2; i < lines.size(); i++)
                {
                    if (lines[i].size() <= 0)
                    {
                        continue;
                    }
                    auto sarray = split_string(lines[i], " ");
                    std::vector<std::string> useful_array;
                    for (size_t j = 0; j < sarray.size(); j++)
                    {
                        if (sarray[j] != std::string(""))
                        {
                            useful_array.push_back(sarray[j]);
                        }
                    }
                    pipe_param_str_.push_back(useful_array);
                }

                for (size_t i = 0; i < pipe_param_str_.size(); i++)
                {
                    operation_param op_param;
                    op_param.type_ = pipe_param_str_[i][0];
                    op_param.name_ = pipe_param_str_[i][1];
                    op_param.device_ = device_;
                    op_param.input_count_ = atoi(pipe_param_str_[i][2].c_str());
                    op_param.output_count_ = atoi(pipe_param_str_[i][3].c_str());
                    if (op_param.output_count_ <= 0)
                    {
                        op_param.output_featmaps_ = std::vector<std::string>();
                    }
                    for (size_t j = 0; j < op_param.output_count_; j++)
                    {
                        op_param.output_featmaps_.push_back(pipe_param_str_[i][4 + op_param.input_count_ + j]);
                    }
                    int specific_start_id = 4 + op_param.input_count_ + op_param.output_count_;
                    for (size_t j = specific_start_id; j < pipe_param_str_[i].size(); j++)
                    {
                        op_param.specific_params_ += (pipe_param_str_[i][j] + " ");
                    }
                    if (op_param.input_count_ <= 0)
                    {
                        // a kind of input method, attach 1 input featmap at top
                        op_param.input_count_ = 1;
                        op_param.input_featmaps_ = std::vector<std::string>{op_param.name_ + "_input"};
                    }
                    else
                    {
                        for (size_t j = 0; j < op_param.input_count_; j++)
                        {
                            op_param.input_featmaps_.push_back(pipe_param_str_[i][4 + j]);
                        }
                    }
                    op_params_.push_back(op_param);
                }

                for (size_t i = 0; i < op_params_.size(); i++)
                {
                    operations_.push_back(operation_reflector<Dtype>::instance().create_object(op_params_[i]));
                }

                // load data
                if (model_file.empty())
                {
                    init_weights();
                }
                else
                {
                    init_weights(model_file);
                }

                //build dag
                op_nodes_.resize(operations_.size());
                for (size_t i = 0; i < operations_.size(); i++)
                {
                    op_nodes_[i] = node<std::string>(operations_[i]->param().name_);
                    operation_names_index_[operations_[i]->param().name_] = i;
                }

                for (size_t i = 0; i < op_params_.size(); i++)
                {
                    for (size_t j = 0; j < op_params_[i].input_count_; j++)
                    {
                        if (featmap_names_index_.find(op_params_[i].input_featmaps_[j]) == featmap_names_index_.end())
                        {
                            featmap_names_index_[op_params_[i].input_featmaps_[j]] = featmap_count_;
                            featmap_count_++;
                        }
                        int id = find_parent_op_index(op_params_[i].input_featmaps_[j]);
                        if (id < 0)
                        {
                            input_featmap_names_.push_back(op_params_[i].input_featmaps_[j]);
                        }
                    }
                    for (size_t j = 0; j < op_params_[i].output_count_; j++)
                    {
                        if (featmap_names_index_.find(op_params_[i].output_featmaps_[j]) == featmap_names_index_.end())
                        {
                            featmap_names_index_[op_params_[i].output_featmaps_[j]] = featmap_count_;
                            featmap_count_++;
                        }
                        int id = find_child_op_index(op_params_[i].output_featmaps_[j]);
                        if (id >= 0)
                        {
                            op_nodes_[i].add_child(op_nodes_[id]);
                        }
                        else
                        {
                            output_featmap_names_.push_back(op_params_[i].output_featmaps_[j]);
                        }
                    }
                }

                featmaps_.resize(featmap_count_);

                CHECK_EQ(input_featmap_names_.size(), 1) << "Now only support 1 input!";
                for (size_t i = 0; i < input_featmap_names_.size(); i++)
                {
                    /*auto id = find_child_op_index(input_featmap_names_[i]);
					auto op_node_vec = bfs_ops_.traverse_undirected(op_nodes_[id]);*/
                    // TODO: Add DAG expand support!!!!
                    for (size_t j = 0; j < op_nodes_.size(); j++)
                    {
                        auto res = operation_names_index_.find(op_nodes_[j].value());
                        if (res == operation_names_index_.end())
                        {
                            LOG(FATAL) << "Un-inited operation.";
                        }
                        ops_execution_order_.push_back(res->second);
                        ops_io_featmap_.push_back(std::pair<std::vector<int>, std::vector<int>>(get_op_input_featmap_idx(res->first), get_op_output_featmap_idx(res->first)));
                    }
                }
            }

            ~impl()
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
            }

            std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>> forward(const std::shared_ptr<memory::tensor<Dtype>> &input_tensor)
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

            std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>> forward_cpu(const std::shared_ptr<memory::tensor<Dtype>> &input_tensor)
            {
                profiler *p = profiler::get();
                if (profile_)
                {
                    p->turn_on();
                    p->scope_start(name_.c_str());
                }
                featmaps_[featmap_names_index_[input_featmap_names_[0]]] = input_tensor;
                for (size_t i = 0; i < ops_execution_order_.size(); i++)
                {
                    p->scope_start(operations_[ops_execution_order_[i]]->param().name_.c_str());
                    std::vector<std::shared_ptr<memory::tensor<Dtype>>> input(ops_io_featmap_[i].first.size());
                    for (size_t j = 0; j < input.size(); j++)
                    {
                        input[j] = featmaps_[ops_io_featmap_[i].first[j]];
                    }
                    std::vector<std::shared_ptr<memory::tensor<Dtype>>> output(ops_io_featmap_[i].second.size());
                    for (size_t j = 0; j < output.size(); j++)
                    {
                        output[j] = std::shared_ptr<memory::tensor<Dtype>>(featmaps_[ops_io_featmap_[i].second[j]]);
                    }
                    operations_[ops_execution_order_[i]]->forward_cpu(input, output);
                    for (size_t j = 0; j < output.size(); j++)
                    {
                        featmaps_[ops_io_featmap_[i].second[j]] = output[j];
                    }
                    p->scope_end();
                }
                std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>> results;
                for (size_t i = 0; i < output_featmap_names_.size(); i++)
                {
                    results[output_featmap_names_[i]] = featmaps_[featmap_names_index_[output_featmap_names_[i]]];
                }
                if (profile_)
                {
                    p->scope_end();
                    p->turn_off();
                }
                return results;
            }

            std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>> forward_gpu(const std::shared_ptr<memory::tensor<Dtype>> &input_tensor)
            {
                profiler *p = profiler::get();
                if (profile_)
                {
                    p->turn_on();
                    p->scope_start(name_.c_str());
                }
                featmaps_[featmap_names_index_[input_featmap_names_[0]]] = input_tensor;
                for (size_t i = 0; i < ops_execution_order_.size(); i++)
                {
                    p->scope_start(operations_[ops_execution_order_[i]]->param().name_.c_str());
                    std::vector<std::shared_ptr<memory::tensor<Dtype>>> input(ops_io_featmap_[i].first.size());
                    for (size_t j = 0; j < input.size(); j++)
                    {
                        input[j] = featmaps_[ops_io_featmap_[i].first[j]];
                    }
                    std::vector<std::shared_ptr<memory::tensor<Dtype>>> output(ops_io_featmap_[i].second.size());
                    for (size_t j = 0; j < output.size(); j++)
                    {
                        output[j] = std::shared_ptr<memory::tensor<Dtype>>(featmaps_[ops_io_featmap_[i].second[j]]);
                    }
                    operations_[ops_execution_order_[i]]->forward_gpu(
#ifdef USE_CUDA
                        cublas_handle_,
#ifdef USE_CUDNN
                        cudnn_handle_,
#endif //!USE_CUDNN
#endif //!USE_CUDA
                        input, output);
                    for (size_t j = 0; j < output.size(); j++)
                    {
                        if (output[j] == nullptr)
                        {
                            std::cout << "!!!!";
                        }
                        featmaps_[ops_io_featmap_[i].second[j]] = output[j];
                    }
                    p->scope_end();
                }
                std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>> results;
                for (size_t i = 0; i < output_featmap_names_.size(); i++)
                {
                    results[output_featmap_names_[i]] = featmaps_[featmap_names_index_[output_featmap_names_[i]]];
                }
                if (profile_)
                {
                    p->scope_end();
                    p->turn_off();
                }
                return results;
            }

            std::shared_ptr<memory::tensor<Dtype>> get_featmap(std::string_view featmap_name)
            {
                return featmaps_[featmap_names_index_[std::string(featmap_name)]];
            }

            void enable_profiler()
            {
                profile_ = true;
            }

            void disable_profiler()
            {
                profile_ = false;
            }

        private:
            void init_weights(std::string_view model_file)
            {
                FILE *fp = fopen(model_file.data(), "rb");
                if (!fp)
                {
                    LOG(FATAL) << "Cannot open " << model_file;
                }
                //LOG(INFO) << "[Pipeline weights memory cost list]=====================";
                for (size_t i = 0; i < operations_.size(); i++)
                {
                    int mem = operations_[i]->init_weights(fp);
                    weights_mem_cost_ += mem;
                    //LOG(INFO) << "[Operation]:\t" << operations_[i]->param().name_ << "\t[weights]:\t" << mem << "(B)";
                }
                //LOG(INFO) << "[Pipeline weights memory cost]: \t" << weights_mem_cost_ * 1.0f / 1024 / 1024 << "(MB)";
                CHECK_EQ(fclose(fp), 0) << "Cannot close " << model_file;
            }

            void init_weights()
            {
                //LOG(INFO) << "[Pipeline weights memory cost list]=====================";
                LOG(WARNING) << "THIS PIPELINE IS LOADED WITH DUMMY DATA.";
                for (size_t i = 0; i < operations_.size(); i++)
                {
                    int mem = operations_[i]->init_weights();
                    weights_mem_cost_ += mem;
                    //LOG(INFO) << "[Operation]:\t" << operations_[i]->param().name_ << "\t[weights]:\t" << mem << "(B)";
                }
                //LOG(INFO) << "[Pipeline weights memory cost]: \t" << weights_mem_cost_ * 1.0f / 1024 / 1024 << "(MB)";
            }

            int find_parent_op_index(std::string inputfeatmap)
            {
                for (size_t i = 0; i < op_params_.size(); i++)
                {
                    for (size_t j = 0; j < op_params_[i].output_count_; j++)
                    {
                        if (inputfeatmap == op_params_[i].output_featmaps_[j])
                        {
                            return i;
                        }
                    }
                }
                return -1;
            }

            int find_child_op_index(std::string outputfeatmap)
            {
                for (size_t i = 0; i < op_params_.size(); i++)
                {
                    for (size_t j = 0; j < op_params_[i].input_count_; j++)
                    {
                        if (outputfeatmap == op_params_[i].input_featmaps_[j])
                        {
                            return i;
                        }
                    }
                }
                return -1;
            }

            std::vector<int> get_op_input_featmap_idx(std::string op_name)
            {
                std::vector<int> result;
                auto aa = operation_names_index_[op_name];
                auto in_feats = op_params_[aa].input_featmaps_;
                for (size_t i = 0; i < in_feats.size(); i++)
                {
                    result.push_back(featmap_names_index_[in_feats[i]]);
                }
                return result;
            }

            std::vector<int> get_op_output_featmap_idx(std::string op_name)
            {
                std::vector<int> result;
                auto aa = operation_names_index_[op_name];
                auto in_feats = op_params_[aa].output_featmaps_;
                for (size_t i = 0; i < in_feats.size(); i++)
                {
                    result.push_back(featmap_names_index_[in_feats[i]]);
                }
                return result;
            }

            std::vector<std::string> read_param_file(std::string_view filepath)
            {
                std::vector<std::string> output;
                std::ifstream in{std::string(filepath)};
                std::string temp;
                if (!in.is_open())
                {
                    return output;
                }
                while (std::getline(in, temp))
                {
                    output.push_back(temp);
                }
                in.close();
                return output;
            }

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

            std::map<std::string, std::vector<float>> blob_scale_table;
            std::map<std::string, std::vector<float>> weight_scale_table;

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

            bool profile_ = false;
            int weights_mem_cost_ = 0;
            int featmap_mem_cost_ = 0;

#ifdef USE_CUDA
            cublasHandle_t cublas_handle_ = nullptr;
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle_ = nullptr;
#endif
#endif
        };

        template <typename Dtype>
        pipeline<Dtype>::pipeline() : impl_{new impl}
        {
        }

        template <typename Dtype>
        pipeline<Dtype>::pipeline(std::string_view param_file, std::string_view model_file, int device) : impl_{new impl{param_file, model_file, device}}
        {
        }

        template <typename Dtype>
        pipeline<Dtype>::pipeline(std::string_view param_file, int device) : impl_{new impl{param_file, device}}
        {
        }

        template <typename Dtype>
        pipeline<Dtype>::pipeline(const std::vector<std::string> &hardcode_params, std::string_view model_file, int device) : impl_{new impl{hardcode_params, model_file, device}}
        {
        }

        template <typename Dtype>
        pipeline<Dtype>::~pipeline()
        {
            if (impl_)
            {
                delete impl_;
            }
        }

        template <typename Dtype>
        std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>> pipeline<Dtype>::forward(const std::shared_ptr<memory::tensor<Dtype>> &input_tensor)
        {
            return impl_->forward(input_tensor);
        }

        template <typename Dtype>
        std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>> pipeline<Dtype>::forward_cpu(const std::shared_ptr<memory::tensor<Dtype>> &input_tensor)
        {
            return impl_->forward_cpu(input_tensor);
        }

        template <typename Dtype>
        std::unordered_map<std::string, std::shared_ptr<memory::tensor<Dtype>>> pipeline<Dtype>::forward_gpu(const std::shared_ptr<memory::tensor<Dtype>> &input_tensor)
        {
            return impl_->forward_gpu(input_tensor);
        }

        template <typename Dtype>
        std::shared_ptr<memory::tensor<Dtype>> pipeline<Dtype>::get_featmap(std::string_view featmap_name)
        {
            return impl_->get_featmap(featmap_name);
        }

        template <typename Dtype>
        void pipeline<Dtype>::enable_profiler()
        {
            impl_->enable_profiler();
        }

        template <typename Dtype>
        void pipeline<Dtype>::disable_profiler()
        {
            impl_->disable_profiler();
        }

        INSTANCE_CLASS(pipeline);
    }
}