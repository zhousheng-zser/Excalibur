#include "log.hpp"
#include "fmt/format.h"
#include "filesystem.hpp"
#include "log_config.hpp"

#include <ctime>
#include <array>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <memory>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string_view>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <Windows.h>
#elif defined(__linux__)
#include <pthread.h>
#else
#error "Unsupported platform."
#endif

#ifdef _WIN32
#define localtime_r(a, b) localtime_s(b, a)
#endif

using namespace glasssix::exposing;

namespace glasssix
{
	namespace
	{
		std::uint32_t get_current_thread_id() noexcept
		{
#ifdef _WIN32
			return GetCurrentThreadId();
#else
			return pthread_self();
#endif
		}

		std::time_t get_local_timestamp() noexcept
		{
			return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		}

		std::uint64_t get_local_days_since_1970() noexcept
		{
			return get_local_timestamp() / 86400ULL;
		}

		const tm& get_local_time(std::time_t timestamp) noexcept
		{
			thread_local std::tm local_time;

			return (localtime_r(&timestamp, &local_time), local_time);
		}

		const tm& get_local_time() noexcept
		{
			return get_local_time(get_local_timestamp());
		}
	}
}

namespace glasssix::logging
{
	namespace
	{
		std::mutex mutex;
		thread_local std::optional<source_location> current_debugging_info;

		/// <summary>
		/// Available terminal colors.
		/// </summary>
		enum class terminal_color : std::size_t
		{
			black,
			white,
			red,
			blue,
			green,
			yellow,
			magenta,
			cyan
		};

		/// <summary>
		/// Corrsponding name to each log level.
		/// </summary>
		constexpr std::array log_level_names
		{
			"NONE",
			"DEBUG",
			"INFO",
			"WARNING",
			"ERROR",
			"FATAL"
		};

		/// <summary>
		/// Corresponding foreground colors to each log level.
		/// </summary>
		constexpr std::array log_level_foreground_colors
		{
			terminal_color::white,
			terminal_color::white,
			terminal_color::white,
			terminal_color::yellow,
			terminal_color::red,
			terminal_color::magenta
		};

		/// <summary>
		/// Formats the log text.
		/// </summary>
		/// <param name="level">The log level</param>
		/// <param name="str">The log message</param>
		/// <param name="including_debugging_info">Determines whether to output the debugging information</param>
		/// <returns>The formatted string</returns>
		std::string format_log_text(log_level level, std::string_view message, bool including_debugging_info)
		{
			decltype(auto) time = get_local_time();

			if (including_debugging_info)
			{
				if (!current_debugging_info)
				{
					current_debugging_info = source_location::current();
				}

				return fmt::format(FMT_STRING("[{:04}-{:02}-{:02} {:02}:{:02}:{:02} {:5} {}:{}][{}] {}"),
					1900 + time.tm_year,
					time.tm_mon + 1,
					time.tm_mday,
					time.tm_hour,
					time.tm_min,
					time.tm_sec,
					get_current_thread_id(),
					current_debugging_info->file,
					current_debugging_info->line,
					log_level_names[static_cast<std::size_t>(level)],
					message
				);
			}
			
			return fmt::format(FMT_STRING("[{:04}-{:02}-{:02} {:02}:{:02}:{:02} {:5}][{}] {}"),
				1900 + time.tm_year,
				time.tm_mon + 1,
				time.tm_mday,
				time.tm_hour,
				time.tm_min,
				time.tm_sec,
				get_current_thread_id(),
				log_level_names[static_cast<std::size_t>(level)],
				message
			);
		}

#ifdef _WIN32
		/// <summary>
		/// Converts colors to a Win32 console attribute.
		/// </summary>
		/// <param name="foreground_color">The foreground color</param>
		/// <param name="background_color">The background color</param>
		/// <returns>The Win32 console attribute</returns>
		std::uint16_t to_win32_console_atrribute(terminal_color foreground_color, terminal_color background_color) noexcept
		{
			static constexpr std::array<std::uint16_t, 8> foreground_colors
			{
				0,
				FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,
				FOREGROUND_RED,
				FOREGROUND_BLUE,
				FOREGROUND_GREEN,
				FOREGROUND_RED | FOREGROUND_GREEN,
				FOREGROUND_RED | FOREGROUND_BLUE,
				FOREGROUND_BLUE | FOREGROUND_GREEN
			};

			static constexpr std::array<std::uint16_t, 8> background_colors
			{
				0,
				BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE,
				BACKGROUND_RED,
				BACKGROUND_BLUE,
				BACKGROUND_GREEN,
				BACKGROUND_RED | BACKGROUND_GREEN,
				BACKGROUND_RED | BACKGROUND_BLUE,
				BACKGROUND_BLUE | BACKGROUND_GREEN
			};

			auto converter = [](const auto& colors, terminal_color color) { return static_cast<std::size_t>(color) > colors.size() ? colors.front() : colors[static_cast<std::size_t>(color)]; };

			return converter(foreground_colors, foreground_color) | converter(background_colors, background_color);
		}

		class terminal_color_decorator
		{
		public:
			terminal_color_decorator(terminal_color foreground_color, terminal_color background_color) noexcept
			{
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), to_win32_console_atrribute(foreground_color, background_color) | FOREGROUND_INTENSITY);
			}

			~terminal_color_decorator() noexcept
			{
				std::puts(str_.c_str());
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), to_win32_console_atrribute(terminal_color::white, terminal_color::black));
			}

			std::string& str() noexcept
			{
				return str_;
			}
		private:
			std::string str_;
		};
#else
		/// <summary>
		/// Converts colors to a linux color code.
		/// </summary>
		/// <param name="foreground_color">The foreground color</param>
		/// <param name="background_color">The background color</param>
		/// <returns>The linux color code</returns>
		std::string to_linux_color_code(terminal_color foreground_color, terminal_color background_color)
		{
			static constexpr std::array foreground_colors
			{
				30,
				37,
				31,
				34,
				32,
				33,
				35,
				36
			};

			static constexpr std::array background_colors
			{
				40,
				47,
				41,
				44,
				42,
				43,
				45,
				46
			};

			auto converter = [](const auto& colors, terminal_color color) { return static_cast<std::size_t>(color) > colors.size() ? colors.front() : colors[static_cast<std::size_t>(color)]; };

			return fmt::format(FMT_STRING("\033[{};{}m"), converter(foreground_colors, foreground_color), converter(background_colors, background_color));
		}

		class terminal_color_decorator
		{
		public:
			terminal_color_decorator(terminal_color foreground_color, terminal_color background_color) : str_{ to_linux_color_code(foreground_color, background_color) }
			{
			}

			~terminal_color_decorator()
			{
				str_.append("\033[0m");
				std::puts(str_.c_str());
			}

			std::string& str() noexcept
			{
				return str_;
			}
		private:
			std::string str_;
		};
#endif

		/// <summary>
		/// Manages the rotation of log files.
		/// </summary>
		class log_rotate
		{
		public:
			log_rotate(std::string_view home_directory, std::string_view application_name) noexcept : home_directory_ { home_directory }, application_name_{ application_name }, days_since_1970_{ get_local_days_since_1970() }
			{
			}

			std::ofstream& current()
			{
				auto days_since_1970 = get_local_days_since_1970();

				if (days_since_1970_ == days_since_1970)
				{
				}

				days_since_1970_ == days_since_1970;
			}
		private:
			void rotate()
			{

			}

			std::string home_directory_;
			std::string application_name_;
			std::uint64_t days_since_1970_;
			std::shared_ptr<FILE> output_file_;
		};
	}

	class log_impl : public implements<log_impl, log>
	{
	public:
		log_impl() noexcept
		{
		}

		void init(const param_string& config_path)
		{
			std::error_code code;
			fs::path real_config_path{ to_narrow_string(config_path) };
			auto config = log_config::load_from_file_or_default(to_narrow_string(config_path));

			if (config.enable_file_output)
			{
				if (!fs::exists(real_config_path, code))
				{
					fs::create_directories(real_config_path, code);
				}
			}
		}

		void debug(const param_string& message, bool including_debugging_info) const
		{
			print_utf8<log_level::debug>(message, including_debugging_info);
		}

		void info(const param_string& message, bool including_debugging_info) const
		{
			print_utf8<log_level::info>(message, including_debugging_info);
		}

		void warning(const param_string& message, bool including_debugging_info) const
		{
			print_utf8<log_level::warning>(message, including_debugging_info);
		}

		void error(const param_string& message, bool including_debugging_info) const
		{
			print_utf8<log_level::error>(message, including_debugging_info);
		}

		void fatal(const param_string& message, bool including_debugging_info) const
		{
			print_utf8<log_level::fatal>(message, including_debugging_info);
			std::terminate();
		}

		void set_log_level(log_level level)
		{
			auto config{ *std::atomic_load_explicit(&config_, std::memory_order_acquire) };

			config.level = level;
			std::atomic_store_explicit(&config_, std::make_shared<log_config>(std::move(config)), std::memory_order_release);
		}
	private:
		template<log_level CurrentLevel>
		void print_utf8(utf8_string_view str, bool including_debugging_info) const
		{
			if (auto level = std::atomic_load_explicit(&config_, std::memory_order_acquire)->level; level != log_level::none && CurrentLevel >= level)
			{
				std::scoped_lock lock{ mutex };
				terminal_color_decorator decorator{ log_level_foreground_colors[static_cast<std::size_t>(CurrentLevel)], terminal_color::black };

				decorator.str().append(format_log_text(CurrentLevel, to_narrow_string(str), including_debugging_info));
			}
		}

		std::ofstream output_stream_;
		std::shared_ptr<log_config> config_;
	};
}

namespace glasssix::logging
{
	EXPORT_EXCALIBUR_PRIMITIVES void* G6_ABI_CALL glasssix_add_ref_get_logger_abi()
	{
		static auto instance{ make_as_first<logging::log_impl>() };
		auto abi = get_abi(instance);

		return (static_cast<impl::abi_unknown_object*>(abi)->add_ref(), abi);
	}

	EXPORT_EXCALIBUR_PRIMITIVES void G6_ABI_CALL glasssix_set_log_debugging_info(const char* file, std::int32_t line)
	{
		if (current_debugging_info)
		{
			current_debugging_info->file = file;
			current_debugging_info->line = line;
		}
		else
		{
			current_debugging_info = source_location{ line, file };
		}
	}
}
