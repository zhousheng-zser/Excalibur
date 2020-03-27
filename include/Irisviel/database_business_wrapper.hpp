#pragma once

#include "database_search_result.hpp"
#include "database_feature_observer.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace glasssix
{
	namespace irisviel
	{
		class database_business_wrapper
		{
		public:
			class impl;

			database_business_wrapper(const std::shared_ptr<database_feature_observer>& observer, const std::string& map_file_path, const std::string& cache_path);
			virtual ~database_business_wrapper();
			bool build(bool rebuild);
			void mark_for_deletion() noexcept;
			std::string cache_file_path() const;
			std::vector<std::vector<database_search_result>> search(const float* feature, std::chrono::milliseconds& elapsed_time, int top) const;
			std::vector<std::vector<database_search_result>> search_many(const std::vector<const float*>& features, std::chrono::milliseconds& elapsed_time, int top) const;
		private:
			impl* impl_;
		};
	}
}
