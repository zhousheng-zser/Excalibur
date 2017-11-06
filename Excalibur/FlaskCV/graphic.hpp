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


		int c_;
		int h_;
		int w_;
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
		//
		bool empty() const;
		int count() const;
		int channel() const;
		int height() const;
		int width() const;
		//
		const float* cpu_data() const;
		float* mutable_cpu_data() const;
#ifdef USE_CUDA
		const float* gpu_data() const;
		float* mutable_gpu_data() const;
#endif
	};
}

#endif // _GRAPHIC_HPP_