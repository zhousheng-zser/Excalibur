#pragma once

#include "filesystem.hpp"
#include "database_cache.hpp"
#include "database_record.hpp"

#include <vector>
#include <string>
#include <cstddef>

namespace glasssix
{
	namespace irisviel
	{
		class face_service
		{
		public:
			face_service(int max_items, int dimension, const std::string& database_path, const std::string& cache_path);
			virtual ~face_service() = default;
			void clear();
			std::string database_path() const;
			std::string cache_path() const;
			void load_databases();
			std::vector<database_search_result> search(const float* feature, int top) const;
			void delete_features(const std::vector<std::string>& keys);
			void delete_feature(const std::string& key);
			void add_features(const std::vector<std::shared_ptr<database_record>>& records);
			void add_feature(database_record& record);
			void update(database_record& record);
			void update_more(const std::vector<std::shared_ptr<database_record>>& records);
		private:
			database_cache create_new_database_core(const std::string& path);
			database_cache find_available_database_core(const std::string& key);

			int max_items_;
			int dimension_;
			fs::path cache_path_;
			fs::path database_path_;
			std::vector<database_cache> cache_;
		};
	}
}
