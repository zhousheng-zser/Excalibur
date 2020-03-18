#include "knn_mapping_manager.hpp"

#include <algorithm>

namespace glasssix
{
	namespace irisviel
	{
		knn_mapping_manager::knn_mapping_manager(const std::string& file_path, std::size_t max_items, int dimension) : dimension_{ dimension }, read_line_position_{ 1 }, file_path_{ file_path }
		{
			mapping_ = std::make_shared<memory_mapping>(file_path, (max_items + 1UL) * knn_mapping_data::struct_size(dimension));

			// Get the mapping description header.
			auto header = mapping_->get_from<knn_mapping_header>(0, knn_mapping_header_traits::header_size);
			if (!header)
			{
				throw std::runtime_error{ "Some error occurs when reading the data header." };
			}

			// Steal the local variable.
			header_ = std::move(*header);

			// Data initializations for the first load.
			if (header_.max_items <= 0)
			{
				header_.max_items = static_cast<int>(max_items);
				header_.current_position = 1;
				header_.version = IRISVIEL_VERSION;
				mapping_->write_to(header_, 0, knn_mapping_header_traits::header_size);
			}
		}

		std::vector<std::shared_ptr<knn_mapping_data>> knn_mapping_manager::get_all_data()
		{
			std::vector<std::shared_ptr<knn_mapping_data>> result;

			if (header_.current_position > 0)
			{
				auto data = mapping_->const_data();
				for (int i = 0; i < header_.current_position; i++)
				{
					auto element = knn_mapping_data::create(dimension_);
					mapping_->get_dynamic_buffer_from(i + 1, *element);

					if (element && element->is_active())
					{
						result.emplace_back(element);
					}
				}
			}

			return result;
		}

		bool knn_mapping_manager::emplace_back(knn_mapping_data& data)
		{
			if (header_.current_position >= header_.max_items || contains(data.key()))
			{
				return false;
			}

			int current_position = ++header_.current_position;
			update_current_position_core();
			data.is_active(true);
			mapping_->write_dynamic_buffer_to(current_position, data);

			return true;
		}

		bool knn_mapping_manager::contains(const std::string& key)
		{
			return search_core([&](const knn_mapping_data& item) { return knn_mapping_data::key_equals(key.c_str(), item.key()); }, [](knn_mapping_data& item, int position) {});
		}

		bool knn_mapping_manager::update(knn_mapping_data& data)
		{
			return search_core([&](const knn_mapping_data& item)
				{
					return knn_mapping_data::key_equals(data, item);
				},
				[&](knn_mapping_data& item, int position)
				{
					mapping_->write_dynamic_buffer_to(position, item);
				});
		}

		std::shared_ptr<knn_mapping_data> knn_mapping_manager::read_next()
		{
			return first_or_default_core<knn_mapping_data>([](const knn_mapping_data& item)
				{
					return true;
				},
				[&](knn_mapping_data& item, int position)
				{
					read_line_position_ = position;

					return item.shared();
				}, read_line_position_);
		}

		bool knn_mapping_manager::delete_by_key(const std::string& key)
		{
			return search_core([&](const knn_mapping_data& item)
				{
					return key == item.key();
				},
				[&](knn_mapping_data& item, int position)
				{
					item.is_active(false);
					mapping_->write_dynamic_buffer_to(position, item);
				});
		}

		std::vector<std::shared_ptr<knn_mapping_data>> knn_mapping_manager::select_by_key(const std::string& key)
		{
			return select_core<knn_mapping_data>([&](const knn_mapping_data& item)
				{
					return key == item.key();
				},
				[&](knn_mapping_data& item, int position)
				{
					return item.shared();
				});
		}

		std::shared_ptr<knn_features> knn_mapping_manager::create_features_reference()
		{
			return std::make_shared<knn_features>(std::bind(&knn_mapping_manager::select_feature_entries_core, this), dimension_);
		}

		void knn_mapping_manager::reset_position()
		{
			read_line_position_ = 1;
		}

		const knn_mapping_header& knn_mapping_manager::header() const
		{
			return header_;
		}

		bool knn_mapping_manager::save_changes() const
		{
			return mapping_->save_changes();
		}

		void knn_mapping_manager::delete_file()
		{
			remove(file_path_.c_str());
		}

		void knn_mapping_manager::update_index_file(const std::string& file_path)
		{
			char cache_name[512] = {};

			file_path.copy(cache_name, file_path.size());
			mapping_->write_to_byte_offset(cache_name, offsetof(knn_mapping_header, index_file_name));
		}

		std::string knn_mapping_manager::file_path() const
		{
			return file_path_;
		}

		void knn_mapping_manager::update_current_position_core()
		{
			mapping_->write_to_byte_offset(header_.current_position, knn_mapping_header_traits::current_position_offset);
		}

		bool knn_mapping_manager::search_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<void(knn_mapping_data&, int)>& action, int start_position, bool only_first)
		{
			if (!predicate || !action)
			{
				return false;
			}

			bool success = false;
			int current_position = header_.current_position;

			for (int i = start_position + 1; i <= current_position; i++)
			{
				auto data = knn_mapping_data::create(dimension_);

				mapping_->get_dynamic_buffer_from(i, *data);

				if (data && data->is_active() && predicate(*data))
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

		bool knn_mapping_manager::search_core(const std::function<bool(const knn_mapping_data&)>& predicate, const std::function<void(knn_mapping_data&, int)>& action, bool only_first)
		{
			return search_core(predicate, action, 0, only_first);
		}

		std::vector<const float*> knn_mapping_manager::select_feature_entries_core()
		{
			std::vector<const float*> result;
			int current_position = header_.current_position;

			for (int i = 1; i <= current_position; i++)
			{
				auto data = mapping_->get_raw_buffer_from(i, knn_mapping_data::struct_size(dimension_));
				auto data_ref = knn_mapping_data::create_ref(dimension_, data);

				if (data_ref->is_active())
				{
					result.emplace_back(data_ref->feature());
				}
			}

			return result;
		}
	}
}
