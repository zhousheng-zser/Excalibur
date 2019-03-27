#pragma once

#include "license_blob.hpp"
#include "license_error.hpp"

#include <string>
#include <filesystem>

namespace glasssix
{
	namespace hippogriff
	{
		class aes_provider;

		/// <summary>
		/// Define a license context.
		/// </summary>
		class license_context
		{
		public:
			license_context(const std::string& name);
			virtual ~license_context();

			/// <summary>
			/// Check if the license is valid.
			/// </summary>
			/// <returns>>The cipher text</returns>
			std::string check();
		protected:
			/// <summary>
			/// Get the machine code encoded in hexadecimal characters.
			/// </summary>
			/// <returns>The machine code</returns>
			static std::string get_machine_code_core();
			std::string get_raw_code_core() const;
			void update_file_core(const license_blob& blob);
			license_blob decrypt_code_core(const std::string& code);
			std::string encrypt_blob_core(const license_blob& blob);
		protected:
			std::shared_ptr<aes_provider> aes_;
		protected:
			const std::string machine_code_;
			static const char* internal_key_;
			const std::experimental::filesystem::path root_directory_;
			const std::experimental::filesystem::path license_file_path_;
		};
	}
}
