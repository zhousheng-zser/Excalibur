#include "simple_window.hpp"
namespace glasssix
{
	simple_window::simple_window(int width, int height)
	{
		static bool registered = false;
		static WNDCLASSEX wcex = { sizeof(WNDCLASSEX) };

		if (!registered)
		{
			wcex.style = CS_HREDRAW | CS_VREDRAW;
			wcex.lpfnWndProc = &simple_window::wnd_proc;
			wcex.cbClsExtra = 0;
			wcex.cbWndExtra = sizeof(void*);
			wcex.hInstance = nullptr;
			wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
			wcex.hbrBackground = nullptr;
			wcex.lpszMenuName = nullptr;
			wcex.lpszClassName = "VideoTrackingWndClass";
			wcex.hIcon = nullptr;

			// 注册窗口
			RegisterClassEx(&wcex);

			registered = true;
		}

		// 计算窗口大小
		RECT window_rect = { 0, 0, width, height };
		DWORD window_style = WS_OVERLAPPEDWINDOW | WS_VISIBLE;
		AdjustWindowRect(&window_rect, window_style, FALSE);
		window_rect.right -= window_rect.left;
		window_rect.bottom -= window_rect.top;
		window_rect.left = (GetSystemMetrics(SM_CXFULLSCREEN) - window_rect.right) / 2;
		window_rect.top = (GetSystemMetrics(SM_CYFULLSCREEN) - window_rect.bottom) / 2;

		// 创建窗口
		handle_ = CreateWindowEx(0, wcex.lpszClassName, "hello", window_style, window_rect.left, window_rect.top, window_rect.right, window_rect.bottom, 0, 0, nullptr, this);
	}

	simple_window::~simple_window()
	{
		if (handle_ != nullptr)
		{
			DestroyWindow(handle_);
			handle_ = nullptr;
		}
	}

	HWND simple_window::handle() const
	{
		return handle_;
	}

	void simple_window::message_loop()
	{
		MSG msg;
		
		while (GetMessage(&msg, nullptr, 0, 0))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	LRESULT __stdcall simple_window::wnd_proc(HWND handle, UINT msg, WPARAM w, LPARAM l)
	{
		return DefWindowProc(handle, msg, w, l);
	}
}
