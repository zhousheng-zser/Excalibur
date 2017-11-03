#include "graphic.hpp"

namespace flaskcv
{
	graphic::graphic()
	{
	}


	graphic::~graphic()
	{
		delete data_;
	}

	graphic::graphic(int c): c_(c), h_(1), w_(1), device_(-1)
	{
		data_ = new syncedmem(c, device_);
	}

}
