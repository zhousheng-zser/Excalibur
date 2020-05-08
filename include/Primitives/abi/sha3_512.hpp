#pragma once

#include "meta.hpp"

#include <array>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <type_traits>
#include <string_view>

namespace glasssix::exposing::hashing::sha3_512
{
	namespace details
	{
		/// <summary>
		/// A std::uint64_t is 64-bit which has an identical bit length to parameter w (also known as the length of z coordinate).
		/// The definition is in Section 5.2.
		/// </summary>
		using word_type = std::uint64_t;

		/// <summary>
		/// The number of rounds.
		/// </summary>
		inline constexpr std::size_t round_size = 24;

		/// <summary>
		/// The size of a word, in bits.
		/// </summary>
		inline constexpr std::size_t word_bits = std::numeric_limits<word_type>::digits;

		/// <summary>
		/// A common factor which is commonly used, i.e. number 5.
		/// </summary>
		inline constexpr std::size_t common_factor = 5;

		/// <summary>
		/// The size of a sponge, in words, defined in Section 5.2.
		/// </summary>
		inline constexpr std::size_t sponge_size = 1600 / std::numeric_limits<std::uint8_t>::digits / sizeof(word_type);

		/// <summary>
		/// Represents a state array defined in Section 3.1, of which the data are word-aligned (std::uint64_t).
		/// </summary>
		class state_array
		{
		public:
			/// <summary>
			/// Creates an instance.
			/// </summary>
			constexpr state_array() noexcept : data_{}
			{
			}

			/// <summary>
			/// Gets the size of the state array.
			/// </summary>
			/// <returns>The size</returns>
			constexpr std::size_t size() const noexcept
			{
				return data_.size();
			}

			/// <summary>
			/// Resets the data.
			/// </summary>
			constexpr void reset() noexcept
			{
				data_ = {};
			}

			/// <summary>
			/// Makes access to a word.
			/// </summary>
			/// <param name="x">The x coordinate</param>
			/// <param name="y">The y coordinate</param>
			/// <returns>A reference to the word</returns>
			constexpr word_type& operator()(std::size_t x, std::size_t y) noexcept
			{
				return data_[calculate_index(x, y)];
			}

			/// <summary>
			/// Makes access to a word in read-only mode.
			/// </summary>
			/// <param name="x">The x coordinate</param>
			/// <param name="y">The y coordinate</param>
			/// <returns>A const reference to the word</returns>
			constexpr const word_type& operator()(std::size_t x, std::size_t y) const noexcept
			{
				return data_[calculate_index(x, y)];
			}
		private:
			static constexpr std::size_t calculate_index(std::size_t x, std::size_t y) noexcept
			{
				// A[x, y, z] = S[w(5y + x) + z] defined in Section 3.1.2.
				// The calculation is simplied as follows because values of a lane (values along z coordinate) are combined into a word (std::uint64_t).
				return common_factor * (y % common_factor) + (x % common_factor);
			}

			std::array<word_type, sponge_size> data_;
		};

		/// <summary>
		/// The context of a hash algorithm.
		/// </summary>
		struct hash_context
		{
			std::size_t capacity;
			state_array state;
			state_array immediate_state;
			std::array<word_type, 5> tmp;

			/// <summary>
			/// Resets all states.
			/// </summary>
			constexpr void reset() noexcept
			{
				tmp = {};
				state.reset();
				immediate_state.reset();
			}
		};

		/// <summary>
		/// A step mapping function named ¦È(A) defined in Section 3.2.1.
		/// </summary>
		/// <param name="context">The hash context</param>
		constexpr void step_mapping_theta(hash_context& context) noexcept
		{
			// For all pairs (x, z) such that 0 ¡Ü x < 5 and 0 ¡Ü z < w, let C[x, z] = A[x, 0, z] ¨’ A[x, 1, z] ¨’ A[x, 2, z] ¨’ A[x, 3, z] ¨’ A[x, 4, z].
			// Here a lane (values along z coordinate) is represented as a word (std::uint64_t).
			for (std::size_t x = 0; x < common_factor; x++)
			{
				context.tmp[x] = meta::apply_index_sequence<common_factor>([&](auto... indexes) { return (context.state(x, indexes) ^ ...); });
			}

			// For all pairs (x, z) such that 0 ¡Ü x < 5 and 0 ¡Ü z < w, let D[x, z] = C[(x - 1) mod 5, z] ¨’ C[(x + 1) mod 5, (z - 1) mod w].
			// For all triples(x, y, z) such that 0 ¡Ü x < 5, 0 ¡Ü y < 5, and 0 ¡Ü z < w, let A¡ä[x, y, z] = A[x, y, z] ¨’ D[x, z].
			// Figure 3 in Section 3.2.1 is intuitive and (z - 1) mod w is equivalent to rotating a word to the left by one bit.
			for (std::size_t x = 0; x < common_factor; x++)
			{
				for (std::size_t y = 0; y < common_factor; y++)
				{
					context.state(x, y) ^= context.tmp[(x - 1) % common_factor] ^ meta::rotl(context.tmp[(x + 1) % common_factor], 1);
				}
			}
		}

		/// <summary>
		/// A step mapping function named ¦Ñ(A) defined in Section 3.2.2.
		/// </summary>
		/// <param name="context">The hash context</param>
		constexpr void step_mapping_rho(hash_context& context) noexcept
		{
			// For t from 0 to 23, (t + 1)(t + 2) / 2 mod w.
			// Here it generates a table for lookup at compile-time.
			constexpr auto rotation_bits = []
			{
				state_array result;

				for (std::size_t x = 1, y = 0, t = 0, tmp = 0; t < 23; t++)
				{
					result(x, y) = ((t + 1) * (t + 2) / 2) % word_bits;

					// (x, y) = (y, (2x + 3y) mod 5)
					tmp = y;
					y = (2 * x + 3 * y) % common_factor;
					x = tmp;
				}

				return result;
			}();

			// For all z such that 0 ¡Ü z ¡Ü w, let A'[0, 0, z] = A[0, 0, z].
			// This operation is omitted and combined into the following loop.

			// (x, y) = (1, 0)
			// For t from 0 to 23.
			for (std::size_t x = 0; x < common_factor; x++)
			{
				for (std::size_t y = 0; y < common_factor; y++)
				{
					// For all z such that 0 ¡Ü z < w, let A¡ä[x, y, z] = A[x, y, (z ¨C (t + 1)(t + 2) / 2) mod w].
					// Here a lane (values along z coordinate) is represented as a word (std::uint64_t).
					// (x, y) = (y, (2x + 3y) mod 5)
					context.immediate_state(y, 2 * x + 3 * y) = meta::rotl(context.state(x, y), rotation_bits(x, y));
				}
			}
		}

		/// <summary>
		/// A step mapping function named ¦Ð(A) defined in Section 3.2.3.
		/// </summary>
		/// <param name="context">The hash context</param>
		constexpr void step_mapping_pi(hash_context& context) noexcept
		{
			for (std::size_t x = 0; x < common_factor; x++)
			{
				for (std::size_t y = 0; y < common_factor; y++)
				{
					// For all triples (x, y, z) such that 0 ¡Ü x < 5, 0 ¡Ü y < 5, and 0 ¡Ü z < w, let A¡ä[x, y, z] = A[(x + 3y) mod 5, x, z].
					// Here a lane (values along z coordinate) is represented as a word (std::uint64_t).
					context.state(x, y) = context.immediate_state(x + 3 * y, x);
				}
			}
		}

		/// <summary>
		/// A step mapping function named ¦Ö(A) defined in Section 3.2.4.
		/// </summary>
		/// <param name="context">The hash context</param>
		constexpr auto step_mapping_chi(hash_context& context) noexcept
		{
			for (std::size_t x = 0; x < common_factor; x++)
			{
				for (std::size_t y = 0; y < common_factor; y++)
				{
					// For all triples (x, y, z) such that 0 ¡Ü x < 5, 0 ¡Ü y < 5, and 0 ¡Ü z < w, let A¡ä[x, y, z] = A[x, y, z] ¨’((A[(x + 1) mod 5, y, z] ¨’ 1) ¡¤ A[(x + 2) mod 5, y, z]).
					// Here a lane (values along z coordinate) is represented as a word (std::uint64_t).
					context.immediate_state(x, y) = context.state(x, y) ^ (~context.state(x + 1, y) & context.state(x + 2, y));
				}
			}
		}

		/// <summary>
		/// A helper function for ¦Ó(A) below defined in Section 3.2.5.
		/// </summary>
		/// <param name="number">The number</param>
		/// <returns>The result</returns>
		constexpr std::uint8_t step_mapping_helper_rc(word_type number) noexcept
		{
			constexpr std::uint8_t max_byte = std::numeric_limits<std::uint8_t>::max();

			// If t mod 255 = 0, return 1.
			if (number % max_byte == 0)
			{
				return 1;
			}

			// Let R = 1000'0000.
			// Here all numbers are inverted for convenience.
			std::uint8_t bit = 0;
			word_type result = 0b0000'0001;

			for (std::size_t i = 1; i <= number % max_byte; i++)
			{
				// For i from 1 to t mod 255, let:
				// a.R = 0 || R;
				// b.R[0] = R[0] ¨’ R[8];
				// c.R[4] = R[4] ¨’ R[8];
				// d.R[5] = R[5] ¨’ R[8];
				// e.R[6] = R[6] ¨’ R[8];
				// f.R = Trunc8[R].
				result <<= 1;
				bit = ((result >> 8) & 0x01);

				meta::set_number_bit(result, 0, (result & 0x01) ^ bit);
				meta::set_number_bit(result, 4, ((result >> 4) & 0x01) ^ bit);
				meta::set_number_bit(result, 5, ((result >> 5) & 0x01) ^ bit);
				meta::set_number_bit(result, 6, ((result >> 6) & 0x01) ^ bit);

				result &= 0xFF;
			}

			// Return R[0].
			return static_cast<std::uint8_t>(result & 0x01);
		}

		/// <summary>
		/// A step mapping function named ¦Ó(A) defined in Section 3.2.5.
		/// </summary>
		/// <param name="context">The hash context</param>
		/// <param name="round">The index of the current round</param>
		constexpr void step_mapping_tau(hash_context& context, std::size_t round) noexcept
		{
			constexpr std::size_t log2_word_bits = meta::log2(word_bits);
			constexpr auto func_rc = [](std::size_t round)
			{
				word_type result = 0;

				for (std::size_t i = 0; i < log2_word_bits; i++)
				{
					meta::set_number_bit(result, (1 << i) - 1, step_mapping_helper_rc(i + 7 * round));
				}

				return result;
			};
			
			constexpr auto rc_table = []
			{
				return meta::apply_index_sequence<round_size>([](auto... indexes) { return std::array<word_type, sizeof...(indexes)>{ func_rc(indexes)... }; });
			}();

			context.state = context.immediate_state;
			context.state(0, 0) ^= rc_table[round];
		}
	}
}
