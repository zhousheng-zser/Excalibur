#pragma once

#include "knn_features.hpp"
#include "memory_mapping.hpp"
#include "knn_mapping_data.hpp"
#include "knn_mapping_header.hpp"

#include <atomic>
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
		// Note: the header length is sizeof(knn_mapping_data) including the padding zero.
		// The first record is beginning at sizeof(knn_mapping_data).
		class knn_mapping_manager
		{
		public:
			knn_mapping_manager(const std::string& file_path, std::size_t max_items, int dimensions);

			bool update(knn_mapping_data& data);
			bool emplace_back(knn_mapping_data& data);
			std::shared_ptr<knn_mapping_data> read_next();
			std::vector<std::shared_ptr<knn_mapping_data>> get_all_data();

			// Delete the first item which key equals the specified key.
			bool delete_by_key(const std::string& key);

			// Select all items which key equals the specified key.
			std::vector<std::shared_ptr<knn_mapping_data>> select_by_key(const std::string& key);

			// Create a reference of all the features.
			std::shared_ptr<knn_features> create_features_reference();

			void reset_position();
			const knn_mapping_header& header() const;
			bool save_changes() const;
			void delete_file();
			void update_index_file(const std::string& file_path);
			std::string file_path() const;
		private:
			void update_current_position_core();

			// Implement 'where'.
			bool search_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<void(knn_mapping_data&, int)>& action, int start_position, bool only_first = true);
			bool search_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<void(knn_mapping_data&, int)>& action, bool only_first = true);
			
			// Get all feature entries in the memory mapping file.
			std::vector<const float*> select_feature_entries_core();

			// Implement 'select'.
			template<typename TResult>
			std::vector<std::shared_ptr<TResult>> select_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<std::shared_ptr<TResult>(knn_mapping_data&, int)>& action)
			{
				std::vector<std::shared_ptr<TResult>> result;

				if (!predicate || !action)
				{
					return result;
				}

				search_core(predicate, [&](knn_mapping_data& item, int position)
				{
					result.emplace_back(action(item, position));
				}, false);

				return result;
			}

			// Implement 'first_or_default'.
			template<typename TResult>
			std::shared_ptr<TResult> first_or_default_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<std::shared_ptr<TResult>(knn_mapping_data&, int)>& action, int start_position = 0)
			{
				std::shared_ptr<TResult> result;

				if (!predicate || !action)
				{
					return result;
				}

				search_core(predicate, [&](knn_mapping_data& item, int position)
				{
					result = action(item, position);
				}, start_position);

				return result;
			}

			int dimension_;
			std::string file_path_;
			knn_mapping_header header_;
			std::atomic_int read_line_position_;
			std::shared_ptr<memory_mapping> mapping_;
		};
	}
}
