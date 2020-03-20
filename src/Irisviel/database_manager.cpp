#include "database_manager.hpp"

#include <algorithm>

namespace glasssix
{
	namespace irisviel
	{
		database_manager::database_manager(const std::string& file_path, std::size_t max_items, int dimension) : dimension_{ dimension }, file_path_{ file_path }
		{
			mapping_ = std::make_shared<memory_mapping>(file_path, (max_items + 1UL) * database_record::struct_size(dimension));

			auto header = mapping_->get_from<database_header>(0, database_header_traits::header_size);

			if (!header)
			{
				throw std::runtime_error{ "Some error occurs when reading the data header." };
			}

			header_ = std::move(*header);

			if (header_.max_items <= 0)
			{
				header_.max_items = static_cast<int>(max_items);
				header_.current_position = 1;
				header_.version = IRISVIEL_VERSION;
				mapping_->write_to(header_, 0, database_header_traits::header_size);
			}
		}

		std::vector<std::shared_ptr<database_record>> database_manager::get_all_data()
		{
			std::vector<std::shared_ptr<database_record>> result;

			if (header_.current_position > 0)
			{
				auto data = mapping_->const_data();

				for (int i = 0; i < header_.current_position; i++)
				{
					auto element = database_record::create(dimension_);

					mapping_->get_dynamic_buffer_from(i + 1, *element);

					if (element && element->active())
					{
						result.emplace_back(element);
					}
				}
			}

			return result;
		}

		bool database_manager::add(database_record& data)
		{
			if (full())
			{
				return false;
			}

			data.active(true);
			mapping_->write_dynamic_buffer_to(header_.current_position, data);

			// Writes the new position.
			header_.current_position++;
			mapping_->write_to_byte_offset(header_.current_position, database_header_traits::current_position_offset);

			return true;
		}

		bool database_manager::contains(const std::string& key)
		{
			return search_core(
				[&](const database_record& item) { return database_record::key_equals(key.c_str(), item.key()); },
				[](database_record& item, int position) {},
				true
				) > 0;
		}

		std::size_t database_manager::update(database_record& data)
		{
			return search_core(
				[&](const database_record& item) { return database_record::key_equals(data, item); },
				[&](database_record& item, int position) { mapping_->write_dynamic_buffer_to(position, item); },
				false
				);
		}

		std::size_t database_manager::remove(const std::string& key)
		{
			return search_core(
				[&](const database_record& item) { return database_record::key_equals(key.c_str(), item.key()); },
				[&](database_record& item, int position) {	item.active(false); mapping_->write_dynamic_buffer_to(position, item); },
				false
				);
		}

		bool database_manager::empty() const noexcept
		{
			return header_.current_position <= 1;
		}

		bool database_manager::full() const noexcept
		{
			return header_.current_position > header_.max_items;
		}

		std::shared_ptr<database_feature_observer> database_manager::create_feature_observer()
		{
			return std::make_shared<database_feature_observer>([this]
				{
					std::vector<const float*> result;
					int current_position = header_.current_position;

					for (int i = 1; i < current_position; i++)
					{
						auto data = mapping_->get_raw_buffer_from(i, database_record::struct_size(dimension_));
						auto data_ref = database_record::create_ref(dimension_, data);

						if (data_ref->active())
						{
							result.emplace_back(data_ref->feature());
						}
					}

					return result;
				}, dimension_);
		}

		bool database_manager::save_changes() const
		{
			return mapping_->save_changes();
		}

		void database_manager::update_index_file(const std::string& file_path)
		{
			char cache_name[512] = {};

			file_path.copy(cache_name, file_path.size());
			mapping_->write_to_byte_offset(cache_name, offsetof(database_header, index_file_name));
		}

		std::string database_manager::file_path() const
		{
			return file_path_;
		}

		std::size_t database_manager::search_core(const std::function<bool(const database_record&)>& predicate, const std::function<void(database_record&, int)>& handler, int start_position, bool only_first)
		{
			std::size_t count = 0;
			int current_position = header_.current_position;

			for (int i = start_position + 1; i < current_position; i++)
			{
				auto data = database_record::create(dimension_);

				mapping_->get_dynamic_buffer_from(i, *data);

				if (data && data->active() && predicate(*data))
				{
					handler(*data, i);
					count++;

					if (only_first)
					{
						break;
					}
				}
			}

			return count;
		}

		std::size_t database_manager::search_core(const std::function<bool(const database_record&)>& predicate, const std::function<void(database_record&, int)>& handler, bool only_first)
		{
			return search_core(predicate, handler, 0, only_first);
		}
	}
}
