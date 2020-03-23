#pragma once

#include "dynamic_buffer.hpp"
#include "database_record.hpp"
#include "database_header.hpp"

#include <mutex>
#include <string>
#include <memory>
#include <fstream>
#include <cstddef>

namespace glasssix
{
	namespace irisviel
	{
		class memory_mapping
		{
		public:
			memory_mapping(const std::string& file_path, std::size_t max_size) : file_path_{ file_path }, max_size_{ max_size }
			{
				std::ifstream stream{ file_path, std::ios::binary };

				buffer_.resize(max_size);
				stream.seekg(0, stream.end);
				std::streamoff length = stream.tellg();

				stream.seekg(0, stream.beg);
				stream.read(const_cast<char*>(buffer_.data()), length);

				// Map the whole file into the memory.
				//file_data_ = MapViewOfFileEx(mapping_handle_, FILE_MAP_ALL_ACCESS, 0, 0, 0, nullptr);
				file_data_ = (void*)(buffer_.data());

				if (file_data_ == nullptr)
				{
					throw std::runtime_error{ "Fail to map the file into the memory." };
				}
			}

			~memory_mapping()
			{
				if (file_data_ != nullptr)
				{
					save_changes();
					//UnmapViewOfFile(file_data_);
					file_data_ = nullptr;
				}
			}

			const void* const_data() const
			{
				return static_cast<const void*>(file_data_);
			}

			void* mutable_data() const
			{
				return file_data_;
			}

			template<typename T>
			const T* const_data() const
			{
				return reinterpret_cast<const T*>(const_cast<const void*>(file_data_));
			}

			template<typename T>
			T* mutable_data() const
			{
				return reinterpret_cast<T*>(file_data_);
			}

			std::size_t size()
			{
				return max_size_;
			}

			std::uint8_t* get_raw_buffer_from(std::size_t offset, std::size_t element_size)
			{
				return (offset + 1) * element_size <= max_size_ ? reinterpret_cast<std::uint8_t*>(file_data_) + offset * element_size : nullptr;
			}

			void write_raw_buffer_to(std::size_t offset, const std::uint8_t* buffer, std::size_t element_size)
			{
				if ((offset + 1) * element_size > max_size_)
				{
					throw std::out_of_range{ "The offset is out of range." };
				}

				std::lock_guard<std::mutex> lock{ mutex_ };

				std::memcpy(reinterpret_cast<std::uint8_t*>(file_data_) + offset * element_size, buffer, element_size);
			}

			void get_dynamic_buffer_from(std::size_t offset, dynamic_buffer& buffer)
			{
				if (auto ptr = get_raw_buffer_from(offset, buffer.size()))
				{
					std::memcpy(buffer.data(), ptr, buffer.size());
				}
			}

			void write_dynamic_buffer_to(std::size_t offset, const dynamic_buffer& buffer)
			{
				write_raw_buffer_to(offset, buffer.data(), buffer.size());
			}

			template<typename Structure>
			Structure* get_fixed_entry_from(std::size_t offset)
			{
				return reinterpret_cast<Structure*>(get_raw_buffer_from(offset, sizeof(Structure)));
			}

			template<typename Structure>
			void write_fixed_entry_to(std::size_t offset, const Structure& buffer)
			{
				write_raw_buffer_to(offset, reinterpret_cast<const std::uint8_t*>(&buffer), sizeof(buffer));
			}

			// Flush all bytes to the disk.
			bool save_changes() const
			{
				std::ofstream stream{ file_path_, std::ios::trunc | std::ios::binary };

				if (!stream.is_open())
				{
					return false;
				}

				stream.write(reinterpret_cast<char*>(file_data_), max_size_);
				stream.flush();

#ifdef __linux__
				system("sync");
				system("sync");
#endif

				return true;
				//return FlushViewOfFile(file_data_, 0);
			}
		private:
			void* file_data_;
			std::mutex mutex_;
			std::string buffer_;
			std::string file_path_;
			std::size_t max_size_;
		};
	}
}
