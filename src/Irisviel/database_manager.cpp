#include "database_manager.hpp"
#include "database_header.hpp"
#include "memory_mapping_operator.hpp"

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
			constexpr auto case_insensitive_string_hasher = [](const std::string& value)
			{

			};

			constexpr auto case_insensitive_string_comparer = [](const std::string& left, const std::string& right)
			{
				return left.size() == right.size() && std::equal(std::begin(left), std::end(left), std::begin(right), std::end(right), [](int left, int right) { return left == right || std::tolower(left) == std::tolower(right); });
			};
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
				if (full())
				{
					return false;
				}

				data.active(true);
				mapping_.write_dynamic_buffer(header_.current_position, data);
				record_entries_.emplace(data.key(), header_.current_position);

				// Writes the new position.
				header_.current_position++;
				mapping_.write_element_absolutely(database_header_traits::current_position_offset, header_.current_position);

				return true;
			}

			bool contains(const std::string& key)
			{
				return record_entries_.find(key) != record_entries_.end();
			}

			std::size_t update(database_record& data)
			{
				auto iter = record_entries_.find(data.key());

				if (iter == record_entries_.end())
				{
					return 0;
				}

				mapping_.write_dynamic_buffer(iter->second, data);

				return 1;
			}

			std::size_t remove(const std::string& key)
			{
				return record_entries_.erase(key);
			}

			bool empty() const noexcept
			{
				return header_.current_position <= 1;
			}

			bool full() const noexcept
			{
				return header_.current_position > header_.max_items;
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
				std::size_t index = 0;

				// Synchronizes the data.
				for (auto [key, index] : record_entries_)
				{
					auto entry = mapping_.locate_element_bytes(index, record_size_);

					mapping_.write_element_bytes(index, entry, record_size_);
					index++;
				}

				header_.current_position = static_cast<int>(record_entries_.size());
				mapping_.write_element_absolutely(database_header_traits::current_position_offset, header_.current_position);
				mapping_.save_changes();
			}

			std::string file_path() const
			{
				return mapping_.path();
			}

		private:
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

			template<typename Predicate, typename RecordHandler>
			std::size_t search_core(Predicate&& predicate, RecordHandler&& handler, int start_position, bool only_first)
			{
				std::size_t count = 0;
				auto data = database_record::create(dimension_);

				for (int i = start_position + 1; i < header_.current_position; i++)
				{
					mapping_.get_dynamic_buffer(i, *data);

					if (data && data->active() && std::forward<Predicate>(predicate)(*data))
					{
						std::forward<RecordHandler>(handler)(*data, i);
						count++;

						if (only_first)
						{
							break;
						}
					}
				}

				return count;
			}

			template<typename Predicate, typename RecordHandler>
			std::size_t search_core(Predicate&& predicate, RecordHandler&& handler, bool only_first)
			{
				return search_core(std::forward<Predicate>(predicate), std::forward<RecordHandler>(handler), 0, only_first);
			}

			int dimension_;
			std::size_t record_size_;
			database_header header_;
			memory_mapping_operator mapping_;
			std::unordered_map<std::string, std::size_t> record_entries_;
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

		std::size_t database_manager::update(database_record& record)
		{
			return impl_->update(record);
		}

		std::size_t database_manager::remove(const std::string& key)
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
