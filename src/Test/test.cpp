#include <gtest/gtest.h>

#include <memory>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <thread>
#include <mutex>
#include <fstream>

#include "log.hpp"
#include "log_config.hpp"
#include "D:\Project\Excalibur\src\Primitives\log_config.cpp"

using namespace glasssix;

TEST(log, disk_size_string)
{
	glasssix::logging::disk_size dsizef("15.2MB");
	ASSERT_EQ(dsizef, 0);

	glasssix::logging::disk_size dsize0("");
	ASSERT_EQ(dsize0, 0);
	ASSERT_EQ(std::string(dsize0), std::string("0B"));

	glasssix::logging::disk_size dsize1(1024);
	ASSERT_EQ(dsize1, 1024ULL);
	ASSERT_EQ(std::string(dsize1), std::string("1KB"));

	glasssix::logging::disk_size dsize2("1024");
	ASSERT_EQ(dsize2, 1024ULL);
	ASSERT_EQ(std::string(dsize2), std::string("1KB"));

	glasssix::logging::disk_size dsize3("1024B");
	ASSERT_EQ(dsize3, 1024ULL);
	ASSERT_EQ(std::string(dsize3), std::string("1KB"));

	glasssix::logging::disk_size dsize4("124kb");
	ASSERT_EQ(dsize4, 124ULL * 1024);
	ASSERT_EQ(std::string(dsize4), std::string("124KB"));

	glasssix::logging::disk_size dsize5("124mB");
	ASSERT_EQ(dsize5, 124ULL * 1024 * 1024);
	ASSERT_EQ(std::string(dsize5), std::string("124MB"));

	glasssix::logging::disk_size dsize6("124Gb");
	ASSERT_EQ(dsize6, 124ULL * 1024 * 1024 * 1024);
	ASSERT_EQ(std::string(dsize6), std::string("124GB"));

	glasssix::logging::disk_size dsize7("124TB");
	ASSERT_EQ(dsize7, 124ULL * 1024 * 1024 * 1024 * 1024);
	ASSERT_EQ(std::string(dsize7), std::string("124TB"));
}

TEST(log, config_read_write)
{
	auto json_t = nlohmann::json
	{
		{ "level", 1 },
		{ "max_size", 1},
		{ "enable_file_output", true },
		{ "home_directory", "."},
		{ "application_name", "log_test.exe"}
	};

	auto out_cfg{ glasssix::logging::log_config::default_value() };
	out_cfg.level = glasssix::log_level::info;
	out_cfg.home_directory = "log";
	out_cfg.enable_file_output = true;
	out_cfg.enable_stderr_output = false;
	out_cfg.max_size = glasssix::logging::disk_size("100MB");
	if (std::ofstream ofs{ "log_config.json" })
	{
		ofs << std::setw(4) << nlohmann::json(out_cfg);
	}

	auto read_cfg{ glasssix::logging::log_config::load_from_file_or_default("log_config.json") };

	ASSERT_EQ(out_cfg.level, read_cfg.level);
	ASSERT_EQ(out_cfg.max_size, read_cfg.max_size);
	ASSERT_EQ(out_cfg.enable_file_output, read_cfg.enable_file_output);
	ASSERT_EQ(out_cfg.home_directory, read_cfg.home_directory);
	ASSERT_EQ(out_cfg.application_name, read_cfg.application_name);
}

TEST(log, output)
{
	glasssix::log::init("log_config.json");
	glasssix::log::set_log_level(glasssix::log_level::debug);

	std::vector<std::shared_ptr<std::thread>> threads;
	for (int i = 0; i < 4; ++i)
	{
		threads.push_back(std::make_shared<std::thread>([]
			{
				for (int i = 0; i < 1000000; ++i)
				{
					glasssix::log::d("123456");
					glasssix::log::i("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
					glasssix::log::w(u8"输出中文警告日志测试！");
					glasssix::log::e("Hello");
					glasssix::log::i("Atomic types are types that encapsulate a value whose access is ");
					glasssix::logd::i("guaranteed to not cause data races and can be used to synchronize memory accesses among different threads.");
					//glasssix::logfmt("thread {}")
					//glasssix::logd::f("This header declares two C++ classes.");
				}
			})
		);
	}

	for (auto th : threads)
	{
		if (th->joinable())
		{
			th->join();
		}
	}
}

int main(int argc, char** argv)
{
	printf("%s\n", argv[0]);
	printf("Running main() from %s\n", __FILE__);

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
