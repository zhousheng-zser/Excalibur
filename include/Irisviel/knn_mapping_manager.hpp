#pragma once

#include "knn_features.hpp"
#include "memory_mapping.hpp"
#include "knn_mapping_data.hpp"
#include "knn_mapping_header.hpp"

#include <atomic>
#include <string>
#include <memory>
#include <vector>
#include <boost/optional.hpp>
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
			knn_mapping_manager(const std::string& file_path, size_t max_items) : read_line_position_{ 1 }, file_path_{ file_path }
			{
				mapping_ = std::make_shared<memory_mapping>(file_path, (max_items + 1UL) * sizeof(knn_mapping_data));

				// Get the mapping description header.
				auto header = mapping_->get_from<knn_mapping_header>(0, knn_mapping_header::header_size);
				if (!header)
				{
					throw std::runtime_error{ "Some error occurs when reading the data header." };
				}

				// Steal the local variable.
				header_ = std::move(*header);
				header_.update_function = [this](long long offset, int value) { mapping_->write_to_byte_offset(value, offset); };

				// Data initializations for the first load.
				if (header_.max_items <= 0)
				{
					header_.max_items = static_cast<int>(max_items);
					header_.current_position = 1;
					header_.version = IRISVIEL_VERSION;
					mapping_->write_to(header_, 0, knn_mapping_header::header_size);
				}
			}

			bool update(knn_mapping_data& data);
			bool emplace_back(knn_mapping_data& data);
			boost::optional<knn_mapping_data> read_next();
			std::vector<knn_mapping_data> get_all_data();

			// Delete the first item which key equals the specified key.
			bool delete_by_key(const std::string& key);

			// Select all items which key equals the specified key.
			std::vector<knn_mapping_data> select_by_key(const std::string& key);

			// Create a reference of all the features.
			std::shared_ptr<knn_features> create_features_reference()
			{
				return std::make_shared<knn_features>(std::bind(&knn_mapping_manager::select_feature_entries_core, this), static_cast<int>(sizeof(knn_mapping_data::feature) / sizeof(float)));
			}

			void reset_position()
			{
				read_line_position_ = 1;
			}

			const knn_mapping_header& header() const
			{
				return header_;
			}

			bool save_changes() const
			{
				return mapping_->save_changes();
			}

			void delete_file()
			{
				remove(file_path_.c_str());
			}

			void update_index_file(const std::string& file_path)
			{
				char cache_name[512] = {};
				
				file_path.copy(cache_name, file_path.size());
				mapping_->write_to_byte_offset(cache_name, offsetof(knn_mapping_header, index_file_name));
			}

			std::string file_path() const
			{
				return file_path_;
			}
		private:
			// Implement 'where'.
			bool search_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<void(knn_mapping_data&, int)>& action, int start_position, bool only_first = true)
			{
				if (!predicate || !action)
				{
					return false;
				}

				bool success = false;
				int current_position = header_.current_position;

				for (int i = start_position + 1; i <= current_position; i++)
				{
					auto data = mapping_->get_from<knn_mapping_data>(i);
					if (data && data->is_active && predicate(*data))
					{
						action(*data, i);
						success = true;

						if (only_first)
						{
							break;
						}
					}
				}

				return success;
			}

			bool search_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<void(knn_mapping_data&, int)>& action, bool only_first = true)
			{
				return search_core(predicate, action, 0, only_first);
			}

			// Implement 'select'.
			template<typename TResult>
			std::vector<TResult> select_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<TResult(knn_mapping_data&, int)>& action)
			{
				std::vector<TResult> result;
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
			boost::optional<TResult> first_or_default_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<TResult(knn_mapping_data&, int)>& action, int start_position = 0)
			{
				boost::optional<TResult> result;
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

			// Get all feature entries in the memory mapping file.
			std::vector<const float*> select_feature_entries_core()
			{
				std::vector<const float*> result;
				int current_position = header_.current_position;

				for (int i = 1; i <= current_position; i++)
				{
					auto data = mapping_->get_entry_from<knn_mapping_data>(i);

					if (data->is_active)
					{
						result.emplace_back(data->feature);
					}
				}

				return result;
			}
		private:
			std::string file_path_;
			knn_mapping_header header_;
			std::atomic_int read_line_position_;
			std::shared_ptr<memory_mapping> mapping_;
		};
	}
}
