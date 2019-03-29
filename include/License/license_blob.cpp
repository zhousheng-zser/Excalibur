#include "license_blob.hpp"

namespace glasssix
{
    namespace hippogriff
    {
        license_blob::license_blob() :install_time{}, last_run_time{}, expiring_time{}
        {
        }

        license_blob::license_blob(const json11::Json& json) : license_blob{}
        {
            machine_code = json["machine_code"].string_value();
            install_time = std::strtoll(json["install_time"].string_value().c_str(), nullptr, 10);
            last_run_time = std::strtoll(json["last_run_time"].string_value().c_str(), nullptr, 10);
            expiring_time = std::strtoll(json["expiring_time"].string_value().c_str(), nullptr, 10);
        }

        json11::Json license_blob::to_json() const
        {
            return json11::Json::object
            {
                { "machine_code", machine_code },
                { "install_time", std::to_string(install_time) },
                { "last_run_time", std::to_string(last_run_time) },
                { "expiring_time", std::to_string(expiring_time) }
            };
        }

        bool license_blob::is_valid(const std::string & machine_code) const
        {
            auto time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

            return machine_code == this->machine_code && install_time < time_now && last_run_time <= time_now && expiring_time > time_now && install_time < last_run_time && install_time < expiring_time && last_run_time < expiring_time;
        }

        bool license_blob::is_valid_and_update(const std::string & machine_code)
        {
            if (!is_valid(machine_code))
            {
                return false;
            }

            update_last_run_time();

            return is_valid(machine_code);
        }

        void license_blob::update_last_run_time()
        {
            last_run_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        }
    }
}
