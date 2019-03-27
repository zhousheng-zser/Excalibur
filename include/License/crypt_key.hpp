#pragma once

#include <memory>

#include <Windows.h>
#include <wincrypt.h>

class crypt_hash;
class aes_provider;

/// <summary>
/// Win32 cryptographic encryption algorithms.
/// </summary>
enum class crypt_encryption_algorithm : ALG_ID
{
	AES_128 = CALG_AES_128,
	AES_192 = CALG_AES_192,
	AES_256 = CALG_AES_256
};

/// <summary>
/// Define a cryptographic key
/// </summary>
class crypt_key : public std::enable_shared_from_this<crypt_key>
{
public:
	crypt_key(const std::shared_ptr<aes_provider>& context, const std::shared_ptr<crypt_hash>& hash, crypt_encryption_algorithm algorithm) : context_{ context }, hash_ { hash }
	{
		if (!context)
		{
			throw std::runtime_error{ "The cryptographic provider context cannot be null." };
		}

		if (!hash)
		{
			throw std::runtime_error{ "The hash context cannot be null." };
		}

		if (!CryptDeriveKey(*context, static_cast<ALG_ID>(algorithm), *hash, CRYPT_EXPORTABLE, &key_))
		{
			throw std::runtime_error{ "Failed to derive the key." };
		}
	}

	virtual ~crypt_key()
	{
		if (key_ != 0)
		{
			CryptDestroyKey(key_);
			key_ = 0;
		}
	}

	inline operator HCRYPTKEY() const
	{
		return key_;
	}
private:
	HCRYPTKEY key_;
	std::shared_ptr<crypt_hash> hash_;
	std::shared_ptr<aes_provider> context_;
};
