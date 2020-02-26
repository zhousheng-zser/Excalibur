#include "knn_mapping_manager.hpp"

#include <list>
#include <algorithm>

namespace glasssix
{
	namespace irisviel
	{
		std::vector<knn_mapping_data> knn_mapping_manager::get_all_data()
		{
			std::vector<knn_mapping_data> result;

			if (header_.current_position > 0)
			{
				auto data = mapping_->const_data();
				for (int i = 0; i < header_.current_position; i++)
				{
					auto element = mapping_->get_from<knn_mapping_data>(i + 1);
					if (element && element->is_active)
					{
						result.emplace_back(std::move(*element));
					}
				}
			}

			return result;
		}

		bool knn_mapping_manager::emplace_back(knn_mapping_data& data)
		{
			if (header_.current_position >= header_.max_items)
			{
				return false;
			}

			int current_position = ++header_.current_position;
			header_.update_current_position();
			data.is_active = true;
			mapping_->write_to(data, current_position);

			return true;
		}

		bool knn_mapping_manager::update(knn_mapping_data& data)
		{
			return search_core([&](const knn_mapping_data& item)
			{
				return data.key_equals(item);
			},
				[&](knn_mapping_data& item, int position)
			{
				mapping_->write_to(data, position);
			});
		}

		boost::optional<knn_mapping_data> knn_mapping_manager::read_next()
		{
			return first_or_default_core<knn_mapping_data>([](const knn_mapping_data& item)
			{
				return true;
			},
				[&](knn_mapping_data& item, int position)
			{
				read_line_position_ = position;

				return item;
			}, read_line_position_);
		}

		bool knn_mapping_manager::delete_by_key(const std::string& key)
		{
			return search_core([&](const knn_mapping_data& item)
			{
				return key == item.key;
			},
				[&](knn_mapping_data& item, int position)
			{
				item.is_active = false;
				mapping_->write_to(item, position);
			});
		}

		std::vector<knn_mapping_data> knn_mapping_manager::select_by_key(const std::string& key)
		{
			return select_core<knn_mapping_data>([&](const knn_mapping_data& item)
			{
				return key == item.key;
			},
				[&](knn_mapping_data& item, int position)
			{
				return item;
			});
		}
	}
}
