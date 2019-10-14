#pragma once

#include <cstdint>

#include <library_export_control.hpp>

namespace glasssix
{
	namespace ozymandias
	{
		/// <summary>
		/// A simple UTF-16 string for export.
		/// </summary>
		class SPHINX_LIBRARY_API simple_wstring
		{
		public:
			simple_wstring(const wchar_t* str);
			simple_wstring(const wchar_t* str, size_t size);
			simple_wstring(const simple_wstring& other);
			simple_wstring(simple_wstring&& other);
			simple_wstring& operator=(const simple_wstring& right);
			simple_wstring& operator=(simple_wstring&& right);
			virtual ~simple_wstring();

			/// <summary>
			/// Get the size.
			/// </summary>
			/// <returns>The size</returns>
			size_t size() const;

			/// <summary>
			/// Get the C-style string.
			/// </summary>
			/// <returns>The C-style string</returns>
			const wchar_t* c_str() const;
		private:
			void initialize_core(const wchar_t* str, size_t size);
		private:
			size_t size_;
			wchar_t* buffer_;
		};
	}
}
