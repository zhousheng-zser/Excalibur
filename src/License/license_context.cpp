#include "license_context.h"
#include "aes_provider.hpp"
#include "hardware_info.h"

#include <fstream>

#include <json11.hpp>

namespace filesystem = std::experimental::filesystem;

namespace glasssix
{
	namespace hippogriff
	{
		const char* license_context::internal_key_ = "za6CwMhjCHNSYgpOlwxg8gcXVIwWus0RGbkJWuUcLmcckt9DVYZyCN0JxYhdPBoK9wCwwWrrSKLK51mb0j2MIAEsSRLyDrDAyHUo";

#ifdef _MSC_VER
		license_context::license_context(const std::string& name) : root_directory_{ common::get_all_user_program_data_path() / L"glasssix" / name }, license_file_path_{ root_directory_ / L"license.cfg" }, machine_code_{ get_machine_code_core() }
		{
#elif defined(__GNUC__)
		license_context::license_context(const std::string& name) : root_directory_{ "/etc/glasssix/" + name }, license_file_path_{ root_directory_ / "license.cfg" }, machine_code_{ get_machine_code_core() }
		{
#endif
			// Set all permissions for the directory.
			filesystem::create_directories(root_directory_);
			filesystem::permissions(root_directory_, filesystem::perms::all);

			// Set the AES key and the IV.
			aes_.reset(new aes_provider{});
			aes_->set_key_with_iv(internal_key_, name);
		}

		license_context::~license_context()
		{
		}

		/// <summary>
		/// Check if the license is valid.
		/// </summary>
		/// <returns>>The cipher text</returns>
		std::string license_context::check()
		{
			// Get the cipher text and decrypt it.
			auto cipher_text = get_raw_code_core();
			auto blob = decrypt_code_core(cipher_text);

			// Check if the license is valid and update the "last run time".
			if (!blob.is_valid_and_update(machine_code_))
			{
				throw license_error{ license_error_code::invalid_license };
			}

			// Update the license file.
			update_file_core(blob);

			return cipher_text;
		}

		std::string license_context::get_machine_code_core()
		{
			hardware_info hardware;
			auto machine_code = hardware.machine_code();

			return common::convert_bytes_to_hex_string(reinterpret_cast<const uint8_t*>(machine_code.c_str()), machine_code.size());
		}

		std::string license_context::get_raw_code_core() const
		{
			if (!filesystem::exists(license_file_path_))
			{
				throw license_error{ license_error_code::file_not_exist };
			}

			if (machine_code_.empty())
			{
				throw license_error{ license_error_code::machine_code_failure };
			}

			// Try opening the license file.
			std::fstream stream{ license_file_path_.c_str(), std::ios::binary | std::ios::in };
			if (!stream.is_open())
			{
				throw license_error{ license_error_code::file_open_failure };
			}

			
			// Get the cipher text.
			return std::string{ std::istreambuf_iterator<char>{ stream }, std::istreambuf_iterator<char>{} };
		}

		license_blob license_context::decrypt_code_core(const std::string& code)
		{
			std::string plain_text;

			try
			{
				plain_text = aes_->decrypt(code);
			}
			catch (std::runtime_error&)
			{
				throw license_error{ license_error_code::decryption_failure };
			}

			// Parse the plain text as a JSON DOM.
			std::string error;
			auto json = json11::Json::parse(plain_text, error);

			// Some error occurs when parsing the text.
			if (!json.is_object() || !error.empty())
			{
				throw license_error{ license_error_code::parse_failure };
			}

			return json;
		}

		std::string license_context::encrypt_blob_core(const license_blob& blob)
		{
			return aes_->encrypt(blob.to_json().dump());
		}

		void license_context::update_file_core(const license_blob& blob)
		{
			std::fstream stream{ license_file_path_.c_str(), std::ios::out | std::ios::trunc | std::ios::binary };
			if (!stream.is_open())
			{
				throw license_error{ license_error_code::update_license };
			}

			try
			{
				// Write the cipher text to the file.
				stream << encrypt_blob_core(blob);
			}
			catch (std::runtime_error&)
			{
				throw license_error{ license_error_code::update_license };
			}

			stream.close();
			filesystem::permissions(license_file_path_, filesystem::perms::all);
		}
	}
}
