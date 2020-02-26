#pragma once

#include "knn_mapping_data.hpp"
#include "knn_mapping_header.hpp"

#include <mutex>
#include <string>
#include <memory>
#include <fstream>

#define LODWORD(x) ((DWORD)(x))
#define HIDWORD(x) ((DWORD)(((DWORDLONG)(x) >> 32) & 0xFFFFFFFF))

namespace glasssix
{
	namespace irisviel
	{
		// Providing basic disk-file memory mapping operations.
		class memory_mapping final
		{
		public:
			memory_mapping(const std::string& file_path, size_t max_size) : file_path_{ file_path },
				max_size_{ max_size }
			{
				std::fstream stream{ file_path, std::ios::binary | std::ios::in | std::ios::out };

				buffer_.resize(max_size);
				stream.seekg(0, stream.end);
				std::streamoff length = stream.tellg();

				stream.seekg(0, stream.beg);
				stream.read(const_cast<char*>(buffer_.data()), static_cast<uint32_t>(length));

				stream.close();

				// Map the whole file into the memory.
				//file_data_ = MapViewOfFileEx(mapping_handle_, FILE_MAP_ALL_ACCESS, 0, 0, 0, nullptr);
				file_data_ = (void*)(buffer_.data());

				if (file_data_ == nullptr)
				{
					throw std::runtime_error{ "Fail in mapping the file into the memory." };
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

			long long size()
			{
				return max_size_;
			}

			// Get the entry in the mapping file.
			template<typename TStruct>
			TStruct* get_entry_from(long long offset)
			{
				if ((offset + 1) * sizeof(TStruct) > max_size_)
				{
					return nullptr;
				}

				return reinterpret_cast<TStruct*>(file_data_) + offset;
			}

			// Fulfill the specified struct with data at the offset and fix the size to read.
			template<typename TStruct>
			std::shared_ptr<TStruct> get_from(long long offset, size_t size)
			{
				if (offset * sizeof(TStruct) + size > max_size_)
				{
					return nullptr;
				}

				auto obj = std::make_shared<TStruct>();
				std::memcpy(obj.get(), reinterpret_cast<TStruct*>(file_data_) + offset, size);

				return obj;
			}

			// Fulfill the specified struct with data at the offset.
			template<typename TStruct>
			std::shared_ptr<TStruct> get_from(long long offset)
			{
				return get_from<TStruct>(offset, sizeof(TStruct));
			}

			// Overwrite the data with the structure at the offset and fix the size to write.
			template<typename TStruct>
			void write_to(const TStruct& obj, long long offset, size_t size)
			{
				if ((offset + 1) * size > max_size_)
				{
					throw std::out_of_range{ "The offset is out of range." };
				}

				std::lock_guard<std::mutex> lock{ mutex_ };
				std::memcpy(reinterpret_cast<TStruct*>(file_data_) + offset, &obj, size);
			}

			// Overwrite the data with the structure at the offset.
			template<typename TStruct>
			void write_to(const TStruct& obj, long long offset)
			{
				write_to(obj, offset, sizeof(TStruct));
			}

			// Overwrite the data with the structure at the offset in bytes.
			template<typename TStruct>
			void write_to_byte_offset(const TStruct& obj, long long byte_offset)
			{
				//LOGD("write_to_byte_offset");
				if (byte_offset + sizeof(TStruct) > max_size_)
				{
					throw std::out_of_range{ "The offset is out of range." };
				}

				std::lock_guard<std::mutex> lock{ mutex_ };
				std::memcpy(reinterpret_cast<uint8_t*>(file_data_) + byte_offset, &obj,
					sizeof(TStruct));
			}

			// Flush all bytes to the disk.
			bool save_changes() const
			{
				std::fstream stream{ file_path_, std::ios::binary | std::ios::out };

				if (!stream.is_open())
				{
					return false;
				}

				stream.write(reinterpret_cast<char*>(file_data_), max_size_);
				stream.flush();
				system("sync");
				system("sync");
				return true;
				//return FlushViewOfFile(file_data_, 0);
			}
		private:
			void* file_data_;
			std::string buffer_;
			std::string file_path_;
			std::mutex mutex_;
			size_t max_size_;
		};
	}
}
