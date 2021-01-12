#include "log_config.hpp"
#include "fmt/format.h"
#include "abi/meta.hpp"
#include "floating_point.hpp"
#include "nlohmann/json_extension.hpp"

#include <tuple>
#include <regex>
#include <fstream>
#include <cstddef>
#include <algorithm>
#include <string_view>

#include <filesystem.hpp>

#ifdef _WIN32
#define NOGDI
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#elif !defined(__linux__)
#error "Unsupported platform."
#endif

namespace glasssix::logging
{
	namespace
	{
		using namespace std::literals;

		enum class disk_size_unit : std::uint64_t
		{
			byte = 'B',
			kilobyte = 'K',
			megabyte = 'M',
			gigabyte = 'G',
			terabyte = 'T',
			petabyte = 'P',
			exabyte = 'E',
		};

		constexpr std::array metadata_pairs
		{
			std::pair{ static_cast<std::size_t>(disk_size_unit::byte), std::pair{ "B"sv, 1ULL } },
			std::pair{ static_cast<std::size_t>(disk_size_unit::kilobyte), std::pair{ "KB"sv, 1ULL << 10 } },
			std::pair{ static_cast<std::size_t>(disk_size_unit::megabyte), std::pair{ "MB"sv, 1ULL << 20 } },
			std::pair{ static_cast<std::size_t>(disk_size_unit::gigabyte), std::pair{ "GB"sv, 1ULL << 30 } },
			std::pair{ static_cast<std::size_t>(disk_size_unit::terabyte), std::pair{ "TB"sv, 1ULL << 40 } },
			std::pair{ static_cast<std::size_t>(disk_size_unit::petabyte), std::pair{ "PB"sv, 1ULL << 50 } },
			std::pair{ static_cast<std::size_t>(disk_size_unit::exabyte), std::pair{ "EB"sv, 1ULL << 60 } }
		};

		constexpr std::array disk_size_names = exposing::meta::apply_index_sequence<metadata_pairs.size()>([](auto... indexes) { return std::array{ metadata_pairs[indexes].second.first... }; });

		/// <summary>
		/// Represents the names among different disk sizes.
		/// </summary>
		constexpr auto disk_size_name_map
		{
			[]
			{
				constexpr std::size_t size = static_cast<std::size_t>(std::max_element(metadata_pairs.begin(), metadata_pairs.end())->first) + 1;
				std::array<std::string_view, size> result{};

				for (const auto& item : metadata_pairs)
				{
					result[item.first] = item.second.first;
				}

				return result;
			}()
		};

		/// <summary>
		/// Represents the ratios among different disk sizes.
		/// </summary>
		constexpr auto disk_size_ratio_map
		{
			[]
			{
				constexpr std::size_t size = static_cast<std::size_t>(std::max_element(metadata_pairs.begin(), metadata_pairs.end())->first) + 1;
				std::array<std::uint64_t, size> result{};

				for (const auto& item : metadata_pairs)
				{
					result[item.first] = item.second.second;
				}

				return result;
			}()
		};

		/// <summary>
		/// Stores a disk size.
		/// </summary>
		class disk_size
		{
		public:
			disk_size(std::string_view value) : value_{}, unit_{}
			{
				thread_local std::regex disk_size_pattern{ fmt::format(FMT_STRING(R"(([+-]?\d*\.?\d+)\s*({})?)"), fmt::join(disk_size_names, "|")), std::regex_constants::icase | std::regex_constants::ECMAScript };

				if (std::cmatch matches; std::regex_match(value.data(), value.data() + value.size(), matches, disk_size_pattern))
				{
					double size = std::stod(matches[1]);

					unit_ = matches[2].matched ? static_cast<disk_size_unit>(matches.str().front()) : disk_size_unit::byte;
					value_ = size;
				}
			}

			disk_size(double value, disk_size_unit unit) noexcept : value_{ value }, unit_{ unit }
			{
			}

			disk_size_unit unit() const noexcept
			{
				return unit_;
			}

			std::uint64_t ratio() const noexcept
			{
				return disk_size_ratio_map[static_cast<std::size_t>(unit_)];
			}

			std::string string() const
			{
				return fmt::format(FMT_STRING("{} {}"), value_, disk_size_name_map[static_cast<std::size_t>(unit_)]);
			}

			bool operator==(const disk_size& right) const noexcept
			{
				return spaceship_compare(*this, right) == 0;
			}

			bool operator!=(const disk_size& right) const noexcept
			{
				return spaceship_compare(*this, right) != 0;
			}

			bool operator<(const disk_size& right) const noexcept
			{
				return spaceship_compare(*this, right) < 0;
			}

			bool operator>(const disk_size& right) const noexcept
			{
				return spaceship_compare(*this, right) > 0;
			}

			bool operator>=(const disk_size& right) const noexcept
			{
				return spaceship_compare(*this, right) >= 0;
			}

			bool operator<=(const disk_size& right) const noexcept
			{
				return spaceship_compare(*this, right) <= 0;
			}
		private:
			static int spaceship_compare(const disk_size& left, const disk_size& right) noexcept
			{
				auto&& [real_left, real_right, relative_ratio] = left.ratio() > right.ratio() ? std::forward_as_tuple(left, right, left.ratio() / right.ratio()) : std::forward_as_tuple(right, left, right.ratio() / left.ratio());
				double right_value_at_same_level = real_right.value_ * relative_ratio;

				if (almost_equals(real_left.value_, right_value_at_same_level))
				{
					return 0;
				}

				return real_left.value_ > right_value_at_same_level ? 1 : -1;
			}

			double value_;
			disk_size_unit unit_;
		};

		/// <summary>
		/// Retrieves the current processs name.
		/// </summary>
		/// <returns>The current process name</returns>
		std::string get_current_process_name()
		{
#ifdef _WIN32
			std::uint32_t real_size = 0;
			std::string buffer(MAX_PATH, '\0');

			while ((real_size = GetModuleFileNameA(nullptr, buffer.data(), static_cast<std::uint32_t>(buffer.size()))) != 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER)
			{
				buffer.resize(buffer.size() * 2);
			}

			buffer.resize(real_size);
			buffer.shrink_to_fit();

			return buffer.empty() ? buffer : fs::path{ buffer }.filename().string();
#elif defined(__linux__)
			std::error_code code;
			auto path = fs::read_symlink("/proc/self/exe", code);

			return code ? std::string{} : fs::path{ path }.filename().string();
#endif
		}
	}

	log_config log_config::default_value()
	{
		return log_config{ log_level::debug, "200 MB", true, ".", get_current_process_name() };
	}

	log_config log_config::load_from_file_or_default(std::string_view path)
	{
		try
		{
			if (std::ifstream stream{ std::string{ path }, std::ios::in | std::ios::binary })
			{
				nlohmann::json json;

				return ((stream >> json), json.get<log_config>());
			}

			return log_config::default_value();
		}
		catch (const std::exception&)
		{
			return log_config::default_value();
		}
	}

	void to_json(nlohmann::json& json, const log_config& value)
	{
		json =
		{
			{ "level", value.level },
			{ "limit", value.limit },
			{ "enable_file_output", value.enable_file_output },
			{ "home_directory", value.home_directory },
			{ "application_name", value.application_name }
		};
	}

	void from_json(const nlohmann::json& json, log_config& value)
	{
		json | nlohmann::get_or_default("level", value.level, log_level::debug);
		json | nlohmann::get_or_default("enable_file_output", value.enable_file_output, true);
		json | nlohmann::get_or_default("home_directory", value.home_directory, ".");
		json | nlohmann::get_or_default("application_name", value.application_name, get_current_process_name());
	}
}
