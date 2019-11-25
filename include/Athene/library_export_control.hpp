#pragma once

#ifndef SPHINX_LIBRARY_EXPORT
#define SPHINX_LIBRARY_API __declspec(dllimport)
#else
#define SPHINX_LIBRARY_API __declspec(dllexport)
#endif
