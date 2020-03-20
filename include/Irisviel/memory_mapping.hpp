#pragma once

#include "dynamic_buffer.hpp"
#include "database_record.hpp"
#include "database_header.hpp"

#include <mutex>
#include <string>
#include <memory>
#include <fstream>
#include <cstddef>

#define LODWORD(x) ((DWORD)(x))
#define HIDWORD(x) ((DWORD)(((DWORDLONG)(x) >> 32) & 0xFFFFFFFF))

namespace glasssix
{
	namespace irisviel
	{
		class memory_mapping final
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

			void get_dynamic_buffer_from(std::size_t offset, dynamic_buffer& buffer)
			{
				if ((offset + 1) * buffer.size() <= max_size_)
				{
					std::memcpy(buffer.data(), reinterpret_cast<const std::uint8_t*>(file_data_) + offset * buffer.size(), buffer.size());
				}
			}

			void write_dynamic_buffer_to(std::size_t offset, const dynamic_buffer& buffer)
			{
				if ((offset + 1) * buffer.size() > max_size_)
				{
					throw std::out_of_range{ "The offset is out of range." };
				}

				std::lock_guard<std::mutex> lock{ mutex_ };

				std::memcpy(reinterpret_cast<std::uint8_t*>(file_data_) + offset * buffer.size(), buffer.data(), buffer.size());
			}

			// Get the entry in the mapping file.
			template<typename Structure>
			Structure* get_entry_from(std::size_t offset)
			{
				if ((offset + 1) * sizeof(Structure) > max_size_)
				{
					return nullptr;
				}

				return reinterpret_cast<Structure*>(file_data_) + offset;
			}

			// Fulfill the specified struct with data at the offset and fix the size to read.
			template<typename Structure>
			std::shared_ptr<Structure> get_from(std::size_t offset, std::size_t size)
			{
				if (offset * sizeof(Structure) + size > max_size_)
				{
					return nullptr;
				}

				auto obj = std::make_shared<Structure>();
				std::memcpy(obj.get(), reinterpret_cast<Structure*>(file_data_) + offset, size);

				return obj;
			}

			// Fulfill the specified struct with data at the offset.
			template<typename Structure>
			std::shared_ptr<Structure> get_from(std::size_t offset)
			{
				return get_from<Structure>(offset, sizeof(Structure));
			}

			// Overwrite the data with the structure at the offset and fix the size to write.
			template<typename Structure>
			void write_to(const Structure& obj, std::size_t offset, std::size_t size)
			{
				if ((offset + 1) * size > max_size_)
				{
					throw std::out_of_range{ "The offset is out of range." };
				}

				std::lock_guard<std::mutex> lock{ mutex_ };
				std::memcpy(reinterpret_cast<Structure*>(file_data_) + offset, &obj, size);
			}

			// Overwrite the data with the structure at the offset.
			template<typename Structure>
			void write_to(const Structure& obj, std::size_t offset)
			{
				write_to(obj, offset, sizeof(Structure));
			}

			// Overwrite the data with the structure at the offset in bytes.
			template<typename Structure>
			void write_to_byte_offset(const Structure& obj, std::size_t byte_offset)
			{
				//LOGD("write_to_byte_offset");
				if (byte_offset + sizeof(Structure) > max_size_)
				{
					throw std::out_of_range{ "The offset is out of range." };
				}

				std::lock_guard<std::mutex> lock{ mutex_ };
				std::memcpy(reinterpret_cast<std::uint8_t*>(file_data_) + byte_offset, &obj, sizeof(Structure));
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
