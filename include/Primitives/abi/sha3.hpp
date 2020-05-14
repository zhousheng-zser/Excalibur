#pragma once

#include "meta.hpp"
#include "sha3_details.hpp"

#include <array>
#include <cstdint>
#include <cstddef>

namespace glasssix::exposing::hashing::sha3
{
	/// <summary>
	/// A helper class that digests input data.
	/// </summary>
	template<sha3_type Type>
	class hash_digest
	{
	public:
		using context_type = details::hash_context<Type>;

		/// <summary>
		/// Updates the context with a piece of data.
		/// </summary>
		/// <param name="Size">The size in bytes</param>
		/// <param name="data">The data</param>
		template<std::size_t Size>
		constexpr void update(const std::array<std::uint8_t, Size>& data) noexcept
		{
			update(data.data(), data.size());
		}

		/// <summary>
		/// Updates the context with a piece of data.
		/// </summary>
		/// <param name="data">The data</param>
		/// <param name="size">The size in bytes</param>
		constexpr void update(const std::uint8_t* data, std::size_t size) noexcept
		{
			auto source_ptr = data;
			auto source_end_ptr = data + size;
			auto destination_ptr = context_.block.data() + context_.block_index;
			
			std::size_t real_size = 0;

			// Fills the internal buffer and updates the context if neccessary.
			while (source_ptr < source_end_ptr)
			{
				real_size = std::min<std::size_t>(source_end_ptr - source_ptr, context_.block_remaining_size());
				meta::copy_bytes(destination_ptr, source_ptr, real_size);

				source_ptr += real_size;
				destination_ptr += real_size;
				context_.block_index += real_size;

				// Updates the state when the internal buffer is full.
				if (context_.block_index >= context_type::block_size)
				{
					details::sponge_step_6(context_);
					destination_ptr = context_.block.data();
				}
			}
		}

		/// <summary>
		/// Finalizes the context (pads the data and calculates the final hash value).
		/// </summary>
		constexpr auto finalize() noexcept
		{
			return details::sponge_finalize(context_);
		}
	private:
		context_type context_;
	};
}
