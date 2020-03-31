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
		class database_manager
		{
		public:
			class impl;

			database_manager(const std::string& file_path, std::size_t capacity, int dimension);
			virtual ~database_manager();
			bool contains(const std::string& key);
			bool update(database_record& record);
			bool remove(const std::string& key);
			bool empty() const noexcept;
			bool full() const noexcept;
			bool add(database_record& record);
			void mark_for_deletion() noexcept;
			std::shared_ptr<database_feature_observer> create_feature_observer();
			void save_changes() noexcept;
			std::string file_path() const;
		private:
			impl* impl_;
		};
	}
}
