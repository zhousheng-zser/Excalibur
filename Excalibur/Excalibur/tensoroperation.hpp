#ifndef _TENSOROPERATION_HPP_
#define _TENSOROPERATION_HPP_

#include "tensor.hpp"

namespace excalibur
{
	template <typename Dtype>
	class tensoroperation
	{
#define Get_Index(x, y, offset) (y*offset+x) 
		enum bordertype { BORDER_CONSTANT, BORDER_REPLICATE };
	public:
		tensoroperation();
		~tensoroperation();

		static void bilinear_resize_cpu(std::shared_ptr<tensor<Dtype>> src, std::shared_ptr<tensor<Dtype>>& dst, int new_height, int new_width);
	};
}
#endif // !_TENSOROPERATION_HPP_



