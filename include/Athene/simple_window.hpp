#pragma once

#include <Windows.h>

namespace glasssix
{
	class simple_window
	{
	public:
		simple_window(int width, int height);
		virtual ~simple_window();

		HWND handle() const;
		static void message_loop();
	private:
		static LRESULT __stdcall wnd_proc(HWND handle, UINT msg, WPARAM w, LPARAM l);
	private:
		HWND handle_;
	};
}
