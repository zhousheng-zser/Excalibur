#pragma once
#ifndef _RECT_HPP_
#define _RECT_HPP_

namespace flaskcv
{
	class rect
	{
		int x;
		int y;
		int h;
		int w;

	public:
		rect();
		rect(int x_, int y_, int h_, int w_);
		~rect();

	};
}


#endif