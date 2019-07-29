#include "IrisvianSearch.hpp"
#include "marshal_fx_ex.hpp"

#include <IrisvielSearch.hpp>

namespace glasssix
{
    namespace irisvian
    {
        /// <summary>
        /// Get the version of the library.
        /// </summary>
        System::String^ IrisvianSearch::Version::get()
        {
            return marshal_fx::marshal_as<System::String^>(irisviel::IrisvielSearch::getVersion());
        }

        /// <summary>
        /// Get the cached base data.
        /// </summary>
        IrisvianSearchDataType^ IrisvianSearch::BaseData::get()
        {
            return base_data_;
        }

        /// <summary>
        /// Create an instance.
        /// </summary>
        /// <param name="dimension">The dimension of a feature</param>
        IrisvianSearch::IrisvianSearch(int dimension)
        {
            dimension_ = dimension;
            native_base_data_ = new std::vector<float>{};
            native_base_data_entries_ = new std::vector<const float*>{};

            searcher_ = new irisviel::IrisvielSearch{ dimension_ };
        }

        /// <summary>
        /// Create an instance.
        /// </summary>
        /// <param name="baseData">The feature group</param>
        /// <param name="dimension">The dimension of a feature</param>
        IrisvianSearch::IrisvianSearch(IrisvianSearchDataType^ baseData, int dimension)
        {
            base_data_ = baseData;
            dimension_ = dimension;
            native_base_data_ = new std::vector<float>{};
            native_base_data_entries_ = new std::vector<const float*>{};

            LoadBaseData();
            searcher_ = new irisviel::IrisvielSearch{ native_base_data_entries_, dimension_ };
        }

        /// <summary>
        /// The finalizer.
        /// </summary>
        IrisvianSearch::~IrisvianSearch()
        {
            this->!IrisvianSearch();
        }

        /// <summary>
        /// Build a graph of the cached data.
        /// </summary>
        /// <returns></returns>
        int IrisvianSearch::BuildGraph()
        {
            CheckPointer();

            return searcher_->buildGraph();
        }

        /// <summary>
        /// Build a graph of the input base data.
        /// </summary>
        /// <param name="baseData">The base data</param>
        /// <returns></returns>
        int IrisvianSearch::BuildGraph(IrisvianSearchDataType^ baseData)
        {
            CheckPointer();

            base_data_ = baseData;
            LoadBaseData();

            return searcher_->buildGraph(native_base_data_entries_);
        }

        /// <summary>
        /// Save the graph.
        /// </summary>
        /// <param name="path">The path</param>
        void IrisvianSearch::SaveGraph(System::String^ path)
        {
            CheckPointer();

            return searcher_->saveGraph(marshal_fx::marshal_as<std::string>(path));
        }

        /// <summary>
        /// Save the graph and the cached base data.
        /// </summary>
        /// <param name="graphPath">The graph path</param>
        /// <param name="baseDataPath">The path of the base data</param>
        void IrisvianSearch::SaveGraph(System::String^ graphPath, System::String^ baseDataPath)
        {
            CheckPointer();

            return searcher_->saveGraph(marshal_fx::marshal_as<std::string>(graphPath), marshal_fx::marshal_as<std::string>(baseDataPath));
        }

        /// <summary>
        /// Load a graph.
        /// </summary>
        /// <param name="path">The graph path</param>
        void IrisvianSearch::LoadGraph(System::String^ path)
        {
            CheckPointer();

            searcher_->loadGraph(marshal_fx::marshal_as<std::string>(path));
        }

        /// <summary>
        /// Optimize the graph.
        /// </summary>
        void IrisvianSearch::OptimizeGraph()
        {
            CheckPointer();

            searcher_->optimizeGraph();
        }

        /// <summary>
        /// Search one or more features.
        /// </summary>
        /// <param name="queryData">The data to search</param>
        /// <param name="topK">The top K</param>
        /// <param name="returnSimilarities">The similarities in percent</param>
        /// <returns>The matched indexes</returns>
        IrisvianSearchResultType^ IrisvianSearch::SearchVector(IrisvianSearchDataType^ queryData, System::UInt32 topK, [System::Runtime::InteropServices::OutAttribute] IrisvianSearchSimilaritiesType^% returnSimilarities)
        {
            CheckPointer();

            // Load native query data.
            std::vector<float> native_data;
            std::vector<const float*> native_data_entries;
            LoadData(queryData, native_data, native_data_entries);

            // Search to fetch native results.
            std::vector<std::vector<uint32_t>> native_result;
            std::vector<std::vector<float>> native_similarities;
            searcher_->searchVector(&native_data_entries, topK, native_result, native_similarities);

            // Allocate the result buffers.
            auto result = marshal_fx::marshal_as<IrisvianSearchResultType^>(native_result);
            returnSimilarities = marshal_fx::marshal_as<IrisvianSearchSimilaritiesType^>(native_similarities);

            return result;
        }

        /// <summary>
        /// Save the search result.
        /// </summary>
        /// <param name="path">The path</param>
        /// <param name="result">The result</param>
        void IrisvianSearch::SaveResult(System::String^ path, IrisvianSearchResultType^ result)
        {
            CheckPointer();

            auto native_result = marshal_fx::marshal_as<std::vector<std::vector<uint32_t>>>(result);

            searcher_->saveResult(marshal_fx::marshal_as<std::string>(path), native_result);
        }

        IrisvianSearch::!IrisvianSearch()
        {
            if (searcher_ != nullptr)
            {
                delete searcher_;
                searcher_ = nullptr;
            }

            if (base_data_ != nullptr)
            {
                base_data_ = nullptr;
            }

            if (native_base_data_ != nullptr)
            {
                delete native_base_data_;
                native_base_data_ = nullptr;
            }

            if (native_base_data_entries_ != nullptr)
            {
                delete native_base_data_entries_;
                native_base_data_entries_ = nullptr;
            }
        }

        void IrisvianSearch::CheckPointer()
        {
            if (searcher_ == nullptr)
            {
                throw gcnew System::ObjectDisposedException{ "The object is uninitialized or disposed." };
            }
        }

        void IrisvianSearch::LoadBaseData()
        {
            // Clear the old data.
            std::decay_t<decltype(*native_base_data_)> new_native_base_data;
            std::decay_t<decltype(*native_base_data_entries_)> new_native_base_data_ptr;

            native_base_data_->swap(new_native_base_data);
            native_base_data_entries_->swap(new_native_base_data_ptr);

            // Load data.
            LoadData(base_data_, *native_base_data_, *native_base_data_entries_);
        }

        void IrisvianSearch::LoadData(IrisvianSearchDataType^ data, std::vector<float>& native_data, std::vector<const float*>& native_data_entries)
        {
            for each (auto item in data)
            {
                auto list = safe_cast<System::Collections::Generic::IList<float>^>(item);

                // Verify the dimension.
                if (list->Count == dimension_)
                {
                    cli::pin_ptr<float> pinned = &item[0];
                    float* data = pinned;

                    // Enlarge the buffer.
                    auto old_size = native_data.size();
                    native_data.resize(old_size + list->Count);

                    // Add the feature.
                    auto merged_data = native_data.data() + old_size;
                    memcpy(merged_data, data, list->Count * sizeof(float));
                    native_data_entries.emplace_back(merged_data);
                }
            }
        }
    }
}
