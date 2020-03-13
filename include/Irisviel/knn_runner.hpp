#pragma once

#include "knn_features.hpp"
#include "irisviel_search.hpp"
#include "knn_search_result.hpp"

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace glasssix
{
	namespace irisviel
	{
		// The KNN algorithm runner.
		class knn_runner
		{
		public:
			knn_runner(const std::shared_ptr<knn_features>& features, const std::string& tmp_path);
			virtual ~knn_runner();

			bool build(bool rebuild = false);
			bool build(const std::string& map_file_name, bool rebuild = false);
			std::string index_file_path() const;
			std::vector<std::vector<knn_search_result>> search(const float* feature, std::chrono::milliseconds& elapsed_time, int top);
			std::vector<std::vector<knn_search_result>> search_many(const std::vector<const float*>& features, std::chrono::milliseconds& elapsed_time, int top);
		private:
			bool valid_state_;
			std::string tmp_path_;
			std::string map_file_path_;
			std::string index_file_path_;
			std::vector<const float*> current_data_;
			std::shared_ptr<knn_features> features_;
			std::shared_ptr<glasssix::irisviel::irisviel_search> searcher_;
		};
	}
}
