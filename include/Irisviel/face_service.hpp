#pragma once

#include "database_record.hpp"
#include "database_search_result.hpp"

#include <memory>
#include <vector>
#include <string>

namespace glasssix
{
	namespace irisviel
	{
		class face_service
		{
		public:
			class impl;

			face_service(int max_items, int dimension, const std::string& working_directory);
			virtual ~face_service();
			void clear() noexcept;
			void remove_all() noexcept;
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
			impl* impl_;
		};
	}
}
