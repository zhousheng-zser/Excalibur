#include "IrisvianSearch.hpp"

#include <marshal_fx.hpp>
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
        IrisvianSearchDataType IrisvianSearch::BaseData::get()
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
            native_base_data_ptr_ = new std::vector<const float*>{};

            searcher_ = new irisviel::IrisvielSearch{ dimension_ };
        }

        /// <summary>
        /// Create an instance.
        /// </summary>
        /// <param name="baseData">The feature group</param>
        /// <param name="dimension">The dimension of a feature</param>
        IrisvianSearch::IrisvianSearch(IrisvianSearchDataType baseData, int dimension)
        {
            base_data_ = baseData;
            dimension_ = dimension;
            native_base_data_ = new std::vector<float>{};
            native_base_data_ptr_ = new std::vector<const float*>{};

            LoadBaseData();
            searcher_ = new irisviel::IrisvielSearch{ native_base_data_ptr_, dimension_ };
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
            return 0;
        }

        /// <summary>
        /// Build a graph of the input base data.
        /// </summary>
        /// <param name="baseData">The base data</param>
        /// <returns></returns>
        int IrisvianSearch::BuildGraph(IrisvianSearchDataType baseData)
        {
            return 0;
        }

        /// <summary>
        /// Save the graph.
        /// </summary>
        /// <param name="path">The path</param>
        void IrisvianSearch::SaveGraph(System::String^ path)
        {
            throw gcnew System::NotImplementedException();
        }

        /// <summary>
        /// Save the graph and the cached base data.
        /// </summary>
        /// <param name="graphPath">The graph path</param>
        /// <param name="baseDataPath">The path of the base data</param>
        void IrisvianSearch::SaveGraph(System::String^ graphPath, System::String^ baseDataPath)
        {
            throw gcnew System::NotImplementedException();
        }

        /// <summary>
        /// Load a graph.
        /// </summary>
        /// <param name="path">The graph path</param>
        void IrisvianSearch::LoadGraph(System::String^ path)
        {
            throw gcnew System::NotImplementedException();
        }

        /// <summary>
        /// Optimize the graph.
        /// </summary>
        void IrisvianSearch::OptimizeGraph()
        {
            throw gcnew System::NotImplementedException();
        }

        /// <summary>
        /// Search one or more features.
        /// </summary>
        /// <param name="queryData">The data to search</param>
        /// <param name="topK">The top K</param>
        /// <param name="returnSimilarities">The similarities in percent</param>
        /// <returns>The matched indexes</returns>
        IrisvianSearchResultType IrisvianSearch::SearchVector(IrisvianSearchDataType queryData, System::UInt32 topK, [System::Runtime::InteropServices::OutAttribute] cli::array<float, 2> ^ %returnSimilarities)
        {
            throw gcnew System::NotImplementedException();
        }

        /// <summary>
        /// Save the search result.
        /// </summary>
        /// <param name="path">The path</param>
        /// <param name="result">The result</param>
        void IrisvianSearch::SaveResult(System::String^ path, IrisvianSearchResultType result)
        {
            throw gcnew System::NotImplementedException();
        }

        IrisvianSearch::!IrisvianSearch()
        {
            throw gcnew System::NotImplementedException();
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
            CheckPointer();

            // Clear the old data.
            native_base_data_->swap({});
            native_base_data_ptr_->swap({});

            for each (auto item in base_data_)
            {
                auto list = safe_cast<System::Collections::Generic::IList<float>^>(item);

                // Verify the dimension.
                if (list->Count == dimension_)
                {
                    cli::pin_ptr<float> pinned = &item[0];
                    float* data = pinned;

                    // Copy the data.
                    auto old_size = native_base_data_->size();
                    native_base_data_->resize(old_size + list->Count);
                    memcpy(native_base_data_->data + old_size, data, list->Count * sizeof(float));

                    // Add the feature pointer.
                    native_base_data_ptr_->emplace_back(native_base_data_->data + old_size);
                }
            }
        }
    }
}
