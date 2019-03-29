#pragma once

#include "md5.h"
#include "base64_utils.h"

#include <string>
#include <filesystem>
#include <unordered_map>

#define CP_GBK 936
#define STL_NO_ERROR(x) ((x) == 0)
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)

#ifdef _MSC_VER
#include <Windows.h>
#endif

namespace filesystem = std::experimental::filesystem;

namespace glasssix
{
	namespace hippogriff
	{
		/// <summary>
		/// Common workarounds for programmers.
		/// </summary>
		class common final
		{
		public:
			/// <summary>
			/// Convert a buffer to a hexadecimal string.
			/// </summary>
			/// <param name="buffer">The buffer</param>
			/// <param name="size">The size of the buffer</param>
			/// <returns>The hexadecimal string</returns>
			static std::string convert_bytes_to_hex_string(const uint8_t* buffer, size_t size)
			{
				std::string result;

				if (size < 1)
				{
					return result;
				}

				// A hexadecimal digit represents a half byte.
				result.resize(size * 2);

				// Fulfill the result buffer.
				for (size_t i = 0; i < size; i++)
				{
					result[i * 2] = hex_digits_[buffer[i] >> 4];
					result[i * 2 + 1] = hex_digits_[buffer[i] & 0x0F];
				}

				return result;
			}

			/// <summary>
			/// Convert a buffer to a hexadecimal string.
			/// </summary>
			/// <param name="buffer">The buffer</param>
			/// <returns>The hexadecimal string</returns>
			inline static std::string convert_bytes_to_hex_string(const std::vector<uint8_t>& buffer)
			{
				return convert_bytes_to_hex_string(buffer.data(), buffer.size());
			}

			/// <summary>
			/// Convert a buffer to a hexadecimal string.
			/// </summary>
			/// <param name="buffer">The buffer</param>
			/// <returns>The hexadecimal string</returns>
			inline static std::string convert_bytes_to_hex_string(const std::string& buffer)
			{
				return convert_bytes_to_hex_string(reinterpret_cast<const uint8_t*>(buffer.c_str()), buffer.size());
			}

			/// <summary>
			/// Convert a hexadecimal string to bytes.
			/// </summary>
			/// <param name="str">The hexadecimal string</param>
			/// <returns>The raw bytes</returns>
			static std::vector<uint8_t> convert_hex_string_to_bytes(const std::string& str)
			{
				std::vector<uint8_t> result;

				// Must be at least one complete byte (Two hexadecimal digits represent one byte).
				if (str.size() < 2)
				{
					return result;
				}

				// A hexadecimal digit represents a half byte.
				result.resize(str.size() / 2);

				// Fulfill the result buffer.
				for (size_t i = 0; i < result.size(); i++)
				{
					result[i] = (hex_character_to_decimal(str[i * 2]) << 4) + hex_character_to_decimal(str[i * 2 + 1]);
				}

				return result;
			}

			/// <summary>
			/// Convert a hexadecimal character to a decimal integer.
			/// </summary>
			/// <param name="hex">The hexadecimal character</param>
			/// <returns>The decimal integer</returns>
			static int hex_character_to_decimal(char hex)
			{
				// Do the first load.
				if (hex_digits_mapping_.empty())
				{
					int index = 0;
					
					std::for_each(std::cbegin(hex_digits_), std::cend(hex_digits_), [&](char digit)
					{
						// Ignore the trailing null-terminating character.
						if (digit != '\0')
						{
							hex_digits_mapping_[digit] = index++;
						}
					});
				}

				auto item = hex_digits_mapping_.find(hex);

				return item != hex_digits_mapping_.cend() ? item->second : 0;
			}

#ifdef _MSC_VER
            /// <summary>
            /// Get an environment variable.
            /// </summary>
            /// <param name="variable">The name of a variable</param>
            /// <returns>The value of the variable</returns>
            static std::string get_environment_variable(const std::string& name)
            {
                size_t size;
                char* buffer;
                std::string result;

                if (STL_NO_ERROR(_dupenv_s(&buffer, &size, name.c_str())) && buffer != nullptr)
                {
                    result = buffer;
                }

                free(buffer);

                return result;
            }

            /// <summary>
            /// Get %ALLUSERPROGRAMDATA%.
            /// </summary>
            /// <returns>The value of %ALLUSERPROGRAMDATA%</returns>
            inline static std::string get_all_user_program_data()
            {
                return get_environment_variable("ALLUSERSPROFILE");
            }

            /// <summary>
            /// Get %ALLUSERPROGRAMDATA% and make it a path.
            /// </summary>
            /// <returns>The path object</returns>
            inline static filesystem::path get_all_user_program_data_path()
            {
                return get_all_user_program_data();
            }

			/// <summary>
			/// Convert UTF-16 to GBK.
			/// </summary>
			/// <param name="str">The string encoded in UTF-16</param>
			/// <param name="size">The size of the string</param>
			/// <returns>The string encoded in GBK</returns>
			static std::string convert_wide_chars_to_multi_bytes(const wchar_t* str, int size)
			{
				std::string result;
				int size_needed = WideCharToMultiByte(CP_GBK, 0, str, static_cast<int>(size), nullptr, 0, nullptr, nullptr);

				if (size_needed <= 0)
				{
					return result;
				}

				result.resize(size_needed);

				// Convert to std::string encoded in GBK.
				WideCharToMultiByte(CP_GBK, 0, str, size, const_cast<char*>(result.data()), static_cast<int>(result.size()), nullptr, nullptr);

				return result;
			}

			/// <summary>
			/// Convert UTF-16 to GBK.
			/// </summary>
			/// <param name="str">The string encoded in UTF-16</param>
			/// <returns>The string encoded in GBK</returns>
			inline static std::string convert_wide_chars_to_multi_bytes(const std::wstring& str)
			{
				return convert_wide_chars_to_multi_bytes(str.c_str(), static_cast<int>(str.size()));
			}

			/// <summary>
			/// Convert UTF-16 to GBK.
			/// </summary>
			/// <param name="str">The string encoded in UTF-16</param>
			/// <param name="size">The size of the string</param>
			/// <returns>The string encoded in GBK</returns>
			static std::wstring convert_multi_bytes_to_wide_chars(const char* str, int size)
			{
				std::wstring result;
				int size_needed = MultiByteToWideChar(CP_GBK, 0, str, size, nullptr, 0);

				if (size_needed <= 0)
				{
					return result;
				}

				result.resize(size_needed);

				// Convert to std::wstring encoded in UTF-16.
				MultiByteToWideChar(CP_GBK, 0, str, size, const_cast<wchar_t*>(result.data()), static_cast<int>(result.size()));

				return result;
			}

			/// <summary>
			/// Convert UTF-16 to GBK.
			/// </summary>
			/// <param name="str">The string encoded in UTF-16</param>
			/// <returns>The string encoded in GBK</returns>
			inline static std::wstring convert_multi_bytes_to_wide_chars(const std::string& str)
			{
				return convert_multi_bytes_to_wide_chars(str.c_str(), static_cast<int>(str.size()));
			}

            /// <summary>
            /// Calculate the MD5 code of the input data.
            /// </summary>
            /// <param name="data">The input data</param>
            /// <param name="size">The size of the data</param>
            /// <returns>The MD5 bytes</returns>
            static std::vector<uint8_t> calculate_md5(const uint8_t* data, uint32_t size)
            {
                md5_digest_context context;

                md5_init_context(&context);
                md5_update_context(&context, data, size);
                md5_final_context(&context);

                return std::vector<uint8_t> { std::cbegin(context.digest), std::cend(context.digest) };
            }

            /// <summary>
            /// Calculate the MD5 code of the input string.
            /// </summary>
            /// <param name="data">The input string</param>
            /// <returns>The MD5 bytes</returns>
            inline static std::vector<uint8_t> calculate_md5(const std::string& str)
            {
                return calculate_md5(reinterpret_cast<const uint8_t*>(str.c_str()), static_cast<uint32_t>(str.size()));
            }
#endif
			/// <summary>
			/// Show a fatal tip and terminate the process.
			/// </summary>
			inline static void fatal_exit()
			{
#ifdef _MSC_VER
                printf_s("Unauthorized SDK.\n");
                exit(0);
                //FatalAppExitA(0, base64_utils::base64_decode("JXU2MEE4JXU3Njg0JXU0RUE3JXU1NEMxJXU1REYyJXU4RkM3JXU2NzFGJXVGRjBDJXU4RjZGJXU0RUY2JXU1QzA2JXU0RjFBJXU1MTczJXU5NUVEJXUzMDAyJXU0RTNBJXU2QjY0JXU2MjExJXU0RUVDJXU2REYxJXU4ODY4JXU2QjQ5JXU2MTBGJXUzMDAy").c_str());
#else
                exit(0);
#endif
			}
		private:
			static const char hex_digits_[17];
			static std::unordered_map<char, int> hex_digits_mapping_;
		};
	}
}
