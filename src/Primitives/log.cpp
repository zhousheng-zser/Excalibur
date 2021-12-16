#include "log.hpp"
#include "fmt/format.h"
#include "filesystem.hpp"
#include "log_config.hpp"
#include "bounded_blocking_queue.hpp"

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
#include <unistd.h>
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

		std::uint32_t get_current_process_id() noexcept
		{
#ifdef _WIN32
			return GetCurrentProcessId();
#else
			return getpid();
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

				return fmt::format("[{:04}-{:02}-{:02} {:02}:{:02}:{:02} {:5} {}:{}][{}] {}",
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

			return fmt::format("[{:04}-{:02}-{:02} {:02}:{:02}:{:02} {:5}][{}] {}",
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

		class append_file
		{
		public:
			explicit append_file(std::string_view filename)
				: fp_{ nullptr }
				, written_bytes_{ 0 }
			{
				fopen_s(&fp_, filename.data(), "a");
				assert(fp_);
			}

			~append_file()
			{
				std::fclose(fp_);
			}

			void append(std::string_view logline)
			{
				std::size_t written{ 0 };

				while (written < logline.size())
				{
					std::size_t n{ std::fwrite(logline.data() + written, 1, logline.size() - written, fp_) };
					if (n == 0)
					{
						if (int err = std::ferror(fp_); err)
						{
							fprintf(stderr, "Append file failed %d\n", err);
						}
						break;
					}
					written += n;
				}

				std::fwrite("\n", 1, 1, fp_);
#ifdef _WIN32
				written_bytes_ += logline.size() + 2;
#else
				written_bytes_ += logline.size() + 1;
#endif
			}

			void flush()
			{
				std::fflush(fp_);
			}

			std::size_t written_bytes() const
			{
				return written_bytes_;
			}

		private:
			FILE* fp_;
			std::size_t written_bytes_;
		};

		/// <summary>
		/// Manages the rotation of log files.
		/// </summary>
		class log_rotate_file
		{
			constexpr static std::uint64_t seconds_per_day_{ 60 * 60 * 24 };

		public:
			log_rotate_file(std::string_view home_directory, std::string_view application_name, std::uint64_t max_size) noexcept
				: home_directory_{ home_directory },
				application_name_{ application_name },
				roll_size_{ max_size },
				flush_interval_{ 3 },
				days_since_1970_{},
				last_roll_{},
				last_flush_{}
			{
				rotate();
			}

			void append(std::string_view logline)
			{
				output_file_->append(logline);

				if (output_file_->written_bytes() > roll_size_)
				{
					rotate();
				}
				else
				{
					const std::time_t now_time{ get_local_timestamp() };

					if ((now_time / seconds_per_day_) != days_since_1970_)
					{
						rotate();
					}
					else if (now_time - last_flush_ > flush_interval_)
					{
						last_flush_ = now_time;
						output_file_->flush();
					}
				}
			}

			void flush()
			{
				output_file_->flush();
			}

		private:
			bool rotate()
			{
				time_t now_time{ get_local_timestamp() };;
				std::string filename = get_log_file_name(now_time);

				if (now_time > last_roll_)
				{
					last_roll_ = now_time;
					last_flush_ = now_time;
					days_since_1970_ = now_time / seconds_per_day_;

					output_file_ = std::make_unique<append_file>(filename);

					return true;
				}

				return false;
			}

			std::string get_log_file_name(const time_t& now)
			{
				const auto& time{ get_local_time(now) };

				// example.exe.20211208-152443.16600.log
				const std::string& filename = fmt::format("{}.{:04}{:02}{:02}-{:02}{:02}{:02}.{}.log",
					application_name_,
					1900 + time.tm_year,
					time.tm_mon + 1,
					time.tm_mday,
					time.tm_hour,
					time.tm_min,
					time.tm_sec,
					get_current_process_id());

				const auto& path = fs::path(home_directory_) / filename;

				return path.string();
			}

		private:
			const std::string home_directory_;
			const std::string application_name_;
			const std::uint64_t roll_size_;
			const std::int32_t flush_interval_;
			std::uint64_t days_since_1970_;
			time_t last_roll_;
			time_t last_flush_;
			std::unique_ptr<append_file> output_file_;
		};
	}

	class log_impl : public implements<log_impl, log>
	{
	public:
		log_impl() noexcept
			: log_queue_{ 4096 },
			running_{ false }
		{
		}

		~log_impl()
		{
			if (running_)
			{
				running_.store(false);
				if (log_to_file_thread_->joinable())
				{
					log_to_file_thread_->join();
				}
			}
		}

		void init(const param_string& config_path)
		{
			static std::once_flag flag;

			std::call_once(flag, [&]
				{
					auto config = log_config::load_from_file_or_default(to_narrow_string(config_path));

					if (config.enable_file_output)
					{
						std::error_code code;
						if (!fs::exists(config.home_directory, code))
						{
							fs::create_directories(config.home_directory, code);
						}

						log_to_file_thread_ = std::make_unique<std::thread>(&log_impl::worker_loop, this);
					}

					std::atomic_store_explicit(&config_, std::make_shared<log_config>(std::move(config)), std::memory_order_release);
				});
		}

		void debug(const param_string& message, bool including_debugging_info)
		{
			print_utf8<log_level::debug>(message, including_debugging_info);
		}

		void info(const param_string& message, bool including_debugging_info)
		{
			print_utf8<log_level::info>(message, including_debugging_info);
		}

		void warning(const param_string& message, bool including_debugging_info)
		{
			print_utf8<log_level::warning>(message, including_debugging_info);
		}

		void error(const param_string& message, bool including_debugging_info)
		{
			print_utf8<log_level::error>(message, including_debugging_info);
		}

		void fatal(const param_string& message, bool including_debugging_info)
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
		void print_utf8(utf8_string_view str, bool including_debugging_info)
		{
			if (auto config{ *std::atomic_load_explicit(&config_, std::memory_order_acquire) }; config.level != log_level::none && CurrentLevel >= config.level)
			{
				auto logline = format_log_text(CurrentLevel, to_narrow_string(str), including_debugging_info);

				if (config.enable_stderr_output)
				{
					std::scoped_lock lock{ mutex };
					terminal_color_decorator decorator{ log_level_foreground_colors[static_cast<std::size_t>(CurrentLevel)], terminal_color::black };

					decorator.str().append(logline);
				}

				if (config.enable_file_output)
				{
					if (!log_queue_.enqueue_for(std::move(logline), std::chrono::milliseconds{ 500 }))
					{
						printf("overrun_counter:%lld\n", log_queue_.size());
					}
				}
			}
		}

		void worker_loop()
		{
			running_.store(true);
			// initial log rotate file
			auto rotate_file{ std::make_unique<log_rotate_file>(config_->home_directory, config_->application_name, config_->max_size) };

			while (running_)
			{
				std::string incoming_async_msg;

				bool dequeued = log_queue_.dequeue_for(incoming_async_msg, std::chrono::seconds(3));
				if (dequeued)
				{
					rotate_file->append(incoming_async_msg);
				}
			}

			rotate_file->flush();
		}

	private:
		std::shared_ptr<log_config> config_;
		memory::bounded_blocking_queue<std::string> log_queue_;
		std::unique_ptr<std::thread> log_to_file_thread_;
		std::atomic<bool> running_;
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
