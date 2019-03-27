#pragma once

#include <string>
#include <chrono>

#include <json11.hpp>

namespace glasssix
{
    namespace hippogriff
    {
        /// <summary>
        /// The activation data.
        /// </summary>
        struct license_blob
        {
            int64_t install_time;
            int64_t expiring_time;
            int64_t last_run_time;
            std::string machine_code;

            license_blob();
            license_blob(const json11::Json& json);
            json11::Json to_json() const;
            void update_last_run_time();
            bool is_valid(const std::string& machine_code) const;
            bool is_valid_and_update(const std::string& machine_code);
        };
    }
}
