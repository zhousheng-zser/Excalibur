#pragma once

#include "database_record.hpp"
#include "database_feature_observer.hpp"

#include <string>
#include <memory>
#include <vector>
#include <cstddef>

namespace glasssix
{
	namespace irisviel
	{
		// Note: the header length is sizeof(database_record) including the padding zero.
		// The first record is beginning at sizeof(database_record).
		class database_manager
		{
		public:
			class impl;

			database_manager(const std::string& file_path, std::size_t max_items, int dimension);
			virtual ~database_manager();
			bool contains(const std::string& key);
			std::size_t update(database_record& record);
			std::size_t remove(const std::string& key);
			bool empty() const noexcept;
			bool full() const noexcept;
			bool add(database_record& record);
			std::vector<std::shared_ptr<database_record>> get_all_data();
			std::shared_ptr<database_feature_observer> create_feature_observer();
			void save_changes() noexcept;
			std::string file_path() const;
		private:
			impl* impl_;
		};
	}
}
