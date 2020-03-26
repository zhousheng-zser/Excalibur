#include "database_manager.hpp"
#include "database_header.hpp"
#include "memory_mapping_operator.hpp"
#include "Primitives/hash_utils.hpp"

#include <cctype>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace glasssix
{
	namespace irisviel
	{
		namespace
		{
			constexpr int database_header_version = 1000;
		}

		class database_manager::impl
		{
		public:
			impl(const std::string& file_path, std::size_t max_items, int dimension) : dimension_{ dimension }, record_size_{ database_record::record_size(dimension) }, mapping_{ file_path, (max_items + 1UL) * database_record::record_size(dimension) }
			{
				auto header = mapping_.locate_element_absolutely<database_header>(0);

				if (!header)
				{
					throw std::runtime_error{ "Some error occurs when reading the data header." };
				}

				header_ = *header;

				if (header_.max_items <= 0)
				{
					header_.max_items = static_cast<int>(max_items);
					header_.current_position = 1;
					header_.version = database_header_version;
					mapping_.write_element_absolutely(0, header_);
				}

				// Builds the index for the key.
				build_index_core();
			}

			std::vector<std::shared_ptr<database_record>> get_all_data()
			{
				std::vector<std::shared_ptr<database_record>> result;

				if (header_.current_position > 0)
				{
					auto data = mapping_.const_data();

					for (int i = 0; i < header_.current_position; i++)
					{
						auto element = database_record::create(dimension_);

						mapping_.get_dynamic_buffer(i + 1, *element);

						if (element && element->active())
						{
							result.emplace_back(element);
						}
					}
				}

				return result;
			}

			bool add(database_record& data)
			{
				if (full() || contains(data.key()))
				{
					return false;
				}

				// Appends an new item at the end.
				data.active(true);
				mapping_.write_dynamic_buffer(header_.current_position, data);
				record_entries_.emplace(data.key(), header_.current_position);

				// Writes the new position.
				update_current_position_core([](int& position) { position++; });

				return true;
			}

			bool contains(const std::string& key)
			{
				return record_entries_.find(key) != record_entries_.end();
			}

			bool update(database_record& data)
			{
				auto iter = record_entries_.find(data.key());

				if (iter == record_entries_.end())
				{
					return false;
				}

				mapping_.write_dynamic_buffer(iter->second, data);

				return true;
			}

			bool remove(const std::string& key)
			{
				return record_entries_.erase(key) > 0;
			}

			bool empty() const noexcept
			{
				return header_.current_position <= 1 && record_entries_.empty();
			}

			bool full() const noexcept
			{
				return header_.current_position > header_.max_items || record_entries_.size() >= header_.max_items;
			}

			void mark_for_deletion() noexcept
			{
				mapping_.mark_for_deletion();
			}

			std::shared_ptr<database_feature_observer> create_feature_observer()
			{
				return std::make_shared<database_feature_observer>([this]
					{
						std::vector<const float*> result;

						for (auto [key, index] : record_entries_)
						{
							auto entry = mapping_.locate_element_bytes(index, record_size_);
							auto data_ref = database_record::create_ref(dimension_, entry);

							result.emplace_back(data_ref->feature());
						}

						return result;
					}, dimension_);
			}

			void save_changes() noexcept
			{
				int new_index = 1;

				// Synchronizes the data.
				for (auto [key, index] : record_entries_)
				{
					auto entry = mapping_.locate_element_bytes(index, record_size_);

					mapping_.write_element_bytes(new_index++, entry, record_size_);
				}

				update_current_position_core([&](int& position) { position = new_index; });
				mapping_.save_changes();
			}

			std::string file_path() const
			{
				return mapping_.path();
			}
		private:
			template<typename Handler>
			void update_current_position_core(Handler&& handler)
			{
				std::forward<Handler>(handler)(header_.current_position);
				mapping_.write_element_absolutely(database_header_traits::current_position_offset, header_.current_position);
			}

			void build_index_core()
			{
				record_entries_.clear();

				for (int i = 1; i < header_.current_position; i++)
				{
					auto entry = mapping_.locate_element_bytes(i, record_size_);
					auto data_ref = database_record::create_ref(dimension_, entry);

					if (data_ref && data_ref->active())
					{
						record_entries_.emplace(data_ref->key(), i);
					}
				}
			}

			int dimension_;
			database_header header_;
			std::size_t record_size_;
			memory_mapping_operator mapping_;
			std::unordered_map<std::string, int, case_insensitive_string_hash, case_insensitive_string_comparer> record_entries_;
		};

		database_manager::database_manager(const std::string& file_path, std::size_t max_items, int dimension) : impl_{ new impl{ file_path, max_items, dimension } }
		{
		}

		database_manager::~database_manager()
		{
			if (impl_)
			{
				delete impl_;
				impl_ = nullptr;
			}
		}

		bool database_manager::contains(const std::string& key)
		{
			return impl_->contains(key);
		}

		bool database_manager::update(database_record& record)
		{
			return impl_->update(record);
		}

		bool database_manager::remove(const std::string& key)
		{
			return impl_->remove(key);
		}

		bool database_manager::empty() const noexcept
		{
			return impl_->empty();
		}

		bool database_manager::full() const noexcept
		{
			return impl_->full();
		}

		bool database_manager::add(database_record& record)
		{
			return impl_->add(record);
		}

		void database_manager::mark_for_deletion() noexcept
		{
			impl_->mark_for_deletion();
		}

		std::vector<std::shared_ptr<database_record>> database_manager::get_all_data()
		{
			return impl_->get_all_data();
		}

		std::shared_ptr<database_feature_observer> database_manager::create_feature_observer()
		{
			return impl_->create_feature_observer();
		}

		void database_manager::save_changes() noexcept
		{
			impl_->save_changes();
		}

		std::string database_manager::file_path() const
		{
			return impl_->file_path();
		}
	}
}
