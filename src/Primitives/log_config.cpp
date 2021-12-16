#include "log_config.hpp"
#include "nlohmann/json_extension.hpp"

#include <regex>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

#include <filesystem.hpp>
#include <fmt/format.h>

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
		static constexpr auto unit_names = std::array{ "TB", "GB", "MB", "KB", "B" };
		static const std::unordered_map<std::string, std::uint64_t> unit_conversion_factor
		{
			{ "", 1ULL },
			{ "B", 1ULL },
			{ "KB", 1ULL << 10 },
			{ "MB", 1ULL << 20 },
			{ "GB", 1ULL << 30 },
			{ "TB", 1ULL << 40 }
		};

		thread_local std::regex disk_size_pattern{ R"((\d+?)\s*?(B|KB|MB|GB|TB)?)", std::regex_constants::icase | std::regex_constants::ECMAScript };

		class disk_size
		{
		public:
			disk_size(std::string_view value) : size_in_bytes_{}
			{
				if (std::cmatch matches; std::regex_match(value.data(), value.data() + value.size(), matches, disk_size_pattern))
				{
					auto count = matches.size();
					assert(count == 3);
					size_in_bytes_ = std::atoll(matches.str(1).c_str());

					std::string& unit = matches.str(2);
					std::transform(unit.begin(), unit.end(), unit.begin(), ::toupper);
					if (auto iter = unit_conversion_factor.find(unit.data()); iter != unit_conversion_factor.cend())
					{
						size_in_bytes_ *= iter->second;
					}
				}
			}

			disk_size(std::uint64_t size_in_bytes) noexcept : size_in_bytes_{ size_in_bytes }
			{
			}

			inline std::uint64_t value() const
			{
				return size_in_bytes_;
			}

			inline std::string to_string() const
			{
				for (const auto& unit : unit_names)
				{
					if (auto iter = unit_conversion_factor.find(unit); iter != unit_conversion_factor.cend() && size_in_bytes_ >= iter->second)
					{
						return fmt::format("{}{}", size_in_bytes_ / iter->second, unit);
					}
				}

				return fmt::format("{}B", size_in_bytes_);
			}

			operator std::uint64_t() const
			{
				return size_in_bytes_;
			}

			operator std::string() const
			{
				return to_string();
			}

		private:
			std::uint64_t size_in_bytes_;
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

			while ((real_size = GetModuleFileNameA(nullptr, buffer.data(), static_cast<std::uint32_t>(buffer.size())), GetLastError() == ERROR_INSUFFICIENT_BUFFER))
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
		return log_config{ log_level::debug, disk_size("1MB"), true, true, ".", get_current_process_name()};
	}

	log_config log_config::load_from_file_or_default(std::string_view path)
	{
		if (std::ifstream stream{ std::string{ path }, std::ios::in | std::ios::binary }; stream)
		{
			nlohmann::json json;

			return ((stream >> json), json.get<log_config>());
		}

		return log_config::default_value();
	}

	void to_json(nlohmann::json& json, const log_config& value)
	{
		json =
		{
			{ "level", value.level },
			{ "max_size", disk_size(value.max_size).to_string()},
			{ "enable_file_output", value.enable_file_output },
			{ "enable_stderr_output", value.enable_stderr_output },
			{ "home_directory", value.home_directory },
			{ "application_name", value.application_name }
		};
	}

	void from_json(const nlohmann::json& json, log_config& value)
	{
		std::string file_max_size;

		json | nlohmann::get_or_default("level", value.level, log_level::debug);
		json | nlohmann::get_or_default("max_size", file_max_size, "1MB");
		json | nlohmann::get_or_default("enable_file_output", value.enable_file_output, true);
		json | nlohmann::get_or_default("enable_stderr_output", value.enable_stderr_output, true);
		json | nlohmann::get_or_default("home_directory", value.home_directory, ".");
		json | nlohmann::get_or_default("application_name", value.application_name, get_current_process_name());

		value.max_size = disk_size(file_max_size);
	}
}
