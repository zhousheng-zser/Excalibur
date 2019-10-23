#pragma once

#include "sampling_data_info.hpp"
#include "net_camera_error_code.hpp"
#include "rtmp_pusher_error_code.hpp"

#include <library_export_control.hpp>

namespace glasssix
{
	struct bitmap_data_provider;

	namespace hippogriff
	{
		class net_camera_impl;

		/// <summary>
		/// Net camera interface.
		/// </summary>
		class SPHINX_LIBRARY_API net_camera
		{
		public:
			using online_changed_handler_type = void(*)(void* any, bool online);
			using sampling_handler_type = void(*)(void* any, const sampling_data_info& info);
			using disconnection_handler_type = void(*)(void* any, const char* name, const char* what, net_camera_error_code code);
			using pusher_disconnection_handler_type = void(*)(void* any, const char* name, const char* what, rtmp_pusher_error_code code);
		public:
			net_camera(const char* name, const char* uri, int fps, int sampling_fps);
			net_camera(const char* name, const char* uri, int fps, int sampling_fps, int buffer_size);
			virtual ~net_camera();

			operator bitmap_data_provider* ();
			operator bitmap_data_provider& ();
			bool online() const;
			void connect();
			void close();
			int fps() const;
			void fps(int fps);
			void stop_rtmp_push();
			void start_rtmp_push(const char* uri);
			void sampling_handler(sampling_handler_type handler, void* any = nullptr);
			void online_changed_handler(online_changed_handler_type handler, void* any = nullptr);
			void disconnection_handler(disconnection_handler_type handler, void* any = nullptr);
			void pusher_disconnection_handler(pusher_disconnection_handler_type handler, void* any = nullptr);
		private:
			net_camera_impl* impl_;
		};
	}
}
