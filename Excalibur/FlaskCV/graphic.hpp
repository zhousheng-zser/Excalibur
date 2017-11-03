#pragma once
#ifndef _GRAPHIC_HPP_
#define _GRAPHIC_HPP_

#include "../Excalibur/syncedmem.hpp"

using namespace excalibur;

namespace flaskcv
{
	class graphic
	{
		// pointer to the data
		syncedmem* data_;

		int w_;
		int h_;
		int c_;
		int device_;


	public:
		// empty
		graphic();
		// vector
		graphic(int c);
		// matrix
		graphic(int h, int w);
		// image
		graphic(int c, int h, int w);
		//copy
		graphic(const graphic& g);
		// external vector
		graphic(int c, float* data);
		// external matrix
		graphic(int h, int w, float* data);
		// external image
		graphic(int c, int h, int w, float* data);
		// release
		~graphic();
		// assign
		graphic& operator=(const graphic& g);
		// set data
		void fill(float x);
		// deep copy
		graphic clone() const;
	};
}

#endif // _GRAPHIC_HPP_