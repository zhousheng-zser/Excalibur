#pragma once

#include <vector>

namespace glasssix
{
    namespace irisviel
    {
        class IrisvielSearch;
    }

    namespace irisvian
    {
        using IrisvianSearchSimilaritiesType = cli::array<cli::array<float>^>;
        using IrisvianSearchResultType = cli::array<cli::array<System::UInt32>^>;
        using IrisvianSearchDataType = System::Collections::Generic::IList<cli::array<float>^>;

        class IrisvielSearch;

        /// <summary>
        /// A Face Search Interface for .NET Framework.
        /// </summary>
        public ref class IrisvianSearch
        {
        public:
            /// <summary>
            /// Get the version of the library.
            /// </summary>
            static property System::String^ Version
            {
                System::String^ get();
            }

            /// <summary>
            /// Get the loaded base data.
            /// </summary>
            property IrisvianSearchDataType^ BaseData
            {
                IrisvianSearchDataType^ get();
            }
        public:
            /// <summary>
            /// Create an instance.
            /// </summary>
            /// <param name="dimension">The dimension of a feature</param>
            IrisvianSearch(int dimension);

            /// <summary>
            /// Create an instance.
            /// </summary>
            /// <param name="baseData">The feature group</param>
            /// <param name="dimension">The dimension of a feature</param>
            IrisvianSearch(IrisvianSearchDataType^ baseData, int dimension);

            /// <summary>
            /// The finalizer.
            /// </summary>
            ~IrisvianSearch();

            /// <summary>
            /// Build a graph of the loaded base data.
            /// </summary>
            /// <returns>The memory peak during the calculation</returns>
            int BuildGraph();

            /// <summary>
            /// Build a graph of the input base data.
            /// </summary>
            /// <param name="baseData">The base data</param>
            /// <returns>The memory peak during the calculation</returns>
            int BuildGraph(IrisvianSearchDataType^ baseData);

            /// <summary>
            /// Save the graph.
            /// </summary>
            /// <param name="path">The path</param>
            void SaveGraph(System::String^ path);

            /// <summary>
            /// Save the graph and the cached base data.
            /// </summary>
            /// <param name="graphPath">The graph path</param>
            /// <param name="baseDataPath">The path of the base data</param>
            void SaveGraph(System::String^ graphPath, System::String^ baseDataPath);

            /// <summary>
            /// Load a graph.
            /// </summary>
            /// <param name="path">The graph path</param>
            void LoadGraph(System::String^ path);

            /// <summary>
            /// Optimize the graph.
            /// </summary>
            void OptimizeGraph();

            /// <summary>
            /// Search one or more features.
            /// </summary>
            /// <param name="queryData">The data to search</param>
            /// <param name="topK">The top K</param>
            /// <param name="returnSimilarities">The similarities in percent</param>
            /// <returns>The matched indexes</returns>
            IrisvianSearchResultType^ SearchVector(IrisvianSearchDataType^ queryData, System::UInt32 topK, [System::Runtime::InteropServices::OutAttribute] IrisvianSearchSimilaritiesType^% returnSimilarities);

            /// <summary>
            /// Save the search result.
            /// </summary>
            /// <param name="path">The path</param>
            /// <param name="result">The result</param>
            void SaveResult(System::String^ path, IrisvianSearchResultType^ result);
        protected:
            !IrisvianSearch();
            void CheckPointer();
        private:
            void LoadBaseData();
            void LoadData(IrisvianSearchDataType^ data, std::vector<float>& native_data, std::vector<const float*>& native_data_entries);
        private:
            int dimension_;
            IrisvianSearchDataType^ base_data_;
            irisviel::IrisvielSearch* searcher_;
            std::vector<float>* native_base_data_;
            std::vector<const float*>* native_base_data_entries_;
        };
    }
}
