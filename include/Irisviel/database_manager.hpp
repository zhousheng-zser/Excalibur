#pragma once

#include "memory_mapping.hpp"
#include "database_record.hpp"
#include "database_header.hpp"
#include "database_feature_observer.hpp"

#include <string>
#include <memory>
#include <vector>
#include <cstddef>
#include <functional>

#define IRISVIEL_VERSION 1000

namespace glasssix
{
	namespace irisviel
	{
		// Note: the header length is sizeof(database_record) including the padding zero.
		// The first record is beginning at sizeof(database_record).
		class database_manager
		{
		public:
			database_manager(const std::string& file_path, std::size_t max_items, int dimension);
			virtual ~database_manager() = default;
			bool contains(const std::string& key);
			std::size_t update(database_record& record);
			std::size_t remove(const std::string& key);
			bool empty() const noexcept;
			bool full() const noexcept;
			bool add(database_record& record);
			std::vector<std::shared_ptr<database_record>> get_all_data();
			std::shared_ptr<database_feature_observer> create_feature_observer();
			bool save_changes() const;
			void update_index_file(const std::string& file_path);
			std::string file_path() const;
		private:
			void update_current_position_core();
			std::size_t search_core(const std::function<bool(const database_record&)>& predicate, const std::function<void(database_record&, int)>& action, int start_position, bool only_first);
			std::size_t search_core(const std::function<bool(const database_record&)>& predicate, const std::function<void(database_record&, int)>& action, bool only_first);

			int dimension_;
			std::string file_path_;
			database_header header_;
			std::shared_ptr<memory_mapping> mapping_;
		};
	}
}
