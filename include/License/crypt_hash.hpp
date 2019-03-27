#pragma once

#include "crypt_key.hpp"
#include "crypt_context.hpp"

#include <string>
#include <memory>
#include <vector>
#include <stdexcept>

#include <Windows.h>
#include <wincrypt.h>

/// <summary>
/// Win32 cryptographic hash algorithms.
/// </summary>
enum class crypt_hash_algorithm : ALG_ID
{
	MD5 = CALG_MD5,
	SHA1 = CALG_SHA1,
	SHA_256 = CALG_SHA_256,
	SHA_512 = CALG_SHA_512
};

/// <summary>
/// Define a Win32 cryptographic hash context.
/// </summary>
class crypt_hash : public std::enable_shared_from_this<crypt_hash>
{
public:
	crypt_hash(const std::shared_ptr<aes_provider>& context, crypt_hash_algorithm algorithm) : context_{ context }
	{
		if (!context_)
		{
			throw std::runtime_error{ "The cryptographic provider context cannot be null." };
		}

		if (!CryptCreateHash(*context_, static_cast<ALG_ID>(algorithm), 0, 0, &hash_))
		{
			throw std::runtime_error{ "Failed to create the hash context." };
		}
	}

	virtual ~crypt_hash()
	{
		if (hash_ != 0)
		{
			CryptDestroyHash(hash_);
			hash_ = 0;
		}
	}

	inline operator HCRYPTHASH() const
	{
		return hash_;
	}

	/// <summary>
	/// Hash data.
	/// </summary>
	/// <param name="data">Data to hash</param>
	/// <param name="size">The size in bytes</param>
	/// <returns>Success or failure</returns>
	inline bool add(const uint8_t* data, uint32_t size)
	{
		return CryptHashData(hash_, data, size, 0);
	}

	/// <summary>
	/// Hash an string as data.
	/// </summary>
	/// <param name="str">The string to hash</param>
	/// <returns>Success or failure</returns>
	inline bool add(const std::string& str)
	{
		return add(reinterpret_cast<const uint8_t*>(str.c_str()), str.size());
	}

	/// <summary>
	/// Get the hash value as a byte array.
	/// </summary>
	/// <returns>The hash value</returns>
	std::vector<uint8_t> value() const
	{
		DWORD size;
		DWORD bytes_read = sizeof(size);

		if (!CryptGetHashParam(hash_, HP_HASHSIZE, reinterpret_cast<uint8_t*>(&size), &bytes_read, 0))
		{
			throw std::runtime_error{ "Failed to acquire the hash size." };
		}

		// Prepare the buffer.
		std::vector<uint8_t> result;
		result.resize(size);
		bytes_read = result.size();

		if (!CryptGetHashParam(hash_, HP_HASHVAL, result.data(), &bytes_read, 0))
		{
			throw std::runtime_error{ "Failed to acquire the hash value." };
		}

		result.shrink_to_fit();

		return result;
	}

	/// <summary>
	/// Derive the key for encryption.
	/// </summary>
	/// <param name="algorithm">The algorithm for encryption</param>
	/// <returns>The key</returns>
	inline std::shared_ptr<crypt_key> get_key(crypt_encryption_algorithm algorithm) const
	{
		return std::make_shared<crypt_key>(context_, shared_from_this(), algorithm);
	}
private:
	HCRYPTHASH hash_;
	std::shared_ptr<aes_provider> context_;
};
