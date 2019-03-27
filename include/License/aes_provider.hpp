#pragma once

#include "common.hpp"

#include <memory>
#include <string>
#include <vector>

#include <openssl/aes.h>

#define AES_IV_SIZE 16
#define AES_BLOCK_SIZE 16

namespace glasssix
{
	namespace hippogriff
	{
		/// <summary>
		/// Define a Win32 AES cryptographic provider.
		/// </summary>
		class aes_provider
		{
		public:
			virtual ~aes_provider() = default;

			/// <summary>
			/// Set an AES key and an IV for the instance.
			/// </summary>
			/// <param name="key">The user key</param>
			/// <param name="iv">The IV</param>
			inline void set_key_with_iv(const std::string& key, const std::vector<uint8_t>& iv)
			{
				// Assign the IV and the hashed key into internal buffers.
				auto key_hash = std::move(common::calculate_md5(key));
				memcpy(initialization_vector_, iv.data(), sizeof(initialization_vector_));

				AES_set_encrypt_key(key_hash.data(), static_cast<int>(key_hash.size()) * 8, &encyption_key_);
				AES_set_decrypt_key(key_hash.data(), static_cast<int>(key_hash.size()) * 8, &decryption_key_);
			}

			/// <summary>
			/// Set an AES key and an IV for the instance.
			/// The IV will be hashed as a MD5 sequence.
			/// </summary>
			/// <param name="key">The user key</param>
			/// <param name="iv">The IV</param>
			inline void set_key_with_iv(const std::string& key, const std::string& iv)
			{
				auto real_iv = common::calculate_md5(iv);
				set_key_with_iv(key, real_iv);
			}

			/// <summary>
			/// Encrypt a plain text.
			/// </summary>
			/// <param name="plain_text">The plain text to encrypt</param>
			/// <returns>The cipher text interpreted as a hexadecimal string</returns>
			std::string encrypt(const std::string& plain_text)
			{
				std::vector<uint8_t> cipher_bytes;

				// Make the plain text aligned by 16 bytes.
				auto padding_bytes = plain_text.size() % AES_BLOCK_SIZE > 0 ? (AES_BLOCK_SIZE - plain_text.size() % AES_BLOCK_SIZE) : 0;
				cipher_bytes.resize(plain_text.size() + padding_bytes);

				// Initialize the IV buffer.
				uint8_t iv_once[AES_IV_SIZE];
				memcpy(iv_once, initialization_vector_, sizeof(initialization_vector_));

				// Encrypt the plain text.
				AES_cbc_encrypt(reinterpret_cast<const uint8_t*>(plain_text.c_str()), cipher_bytes.data(), plain_text.size(), &encyption_key_, iv_once, AES_ENCRYPT);

				return common::convert_bytes_to_hex_string(cipher_bytes);
			}

			/// <summary>
			/// Decrypt a cipher text.
			/// </summary>
			/// <param name="cipher_text">The cipher text to deceypt</param>
			/// <returns>The plain text</returns>
			std::string decrypt(const std::string& cipher_text)
			{
				std::string plain_text;
				auto cipher_bytes = common::convert_hex_string_to_bytes(cipher_text);
				plain_text.resize(cipher_bytes.size());

				// Initialize the IV buffer.
				uint8_t iv_once[AES_IV_SIZE];
				memcpy(iv_once, initialization_vector_, sizeof(initialization_vector_));

				// Remove padding as required.
				AES_cbc_encrypt(cipher_bytes.data(), reinterpret_cast<uint8_t*>(const_cast<char*>(plain_text.c_str())), cipher_bytes.size(), &decryption_key_, iv_once, AES_DECRYPT);

				return plain_text.substr(0, plain_text.find_last_not_of('\0') + 1);
			}
		private:
			AES_KEY encyption_key_;
			AES_KEY decryption_key_;
			uint8_t initialization_vector_[AES_IV_SIZE];
		};
	}
}
