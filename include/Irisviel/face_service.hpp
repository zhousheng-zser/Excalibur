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

			face_service(int single_database_capacity, int dimension, const std::string& working_directory);
			virtual ~face_service();
			void clear() noexcept;
			void remove_all() noexcept;
			std::string database_directory() const;
			std::string cache_directory() const;
			void load_databases();
			std::vector<database_search_result> search(const float* feature, int top) const;
			void remove(const std::vector<std::string>& keys);
			void remove(const std::string& key);
			void add(const std::vector<std::shared_ptr<database_record>>& records);
			void add(database_record& record);
			void update(database_record& record);
			void update(const std::vector<std::shared_ptr<database_record>>& records);
		private:
			impl* impl_;
		};
	}
}
