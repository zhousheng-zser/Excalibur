#include "log_config.hpp"
#include "nlohmann/json_extension.hpp"

#include <regex>
#include <fstream>
#include <cstdint>

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
		thread_local std::regex disk_size_pattern{ R"((\d+?)\s*?(B|KB|MB|GB|PB|EB)?)", std::regex_constants::icase | std::regex_constants::ECMAScript };

		class disk_size
		{
		public:
			disk_size(std::string_view value)
			{
				if (std::cmatch matches; std::regex_match(value.data(), value.data() + value.size(), matches, disk_size_pattern))
				{
					
				}
			}

			disk_size(std::uint64_t size_in_bytes) noexcept : size_in_bytes_{ size_in_bytes }
			{

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
		return log_config{ log_level::debug, 1, true, ".", get_current_process_name() };
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
