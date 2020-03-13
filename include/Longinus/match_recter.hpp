#ifndef _MATCH_RECTOR_HPP_
#define _MATCH_RECTOR_HPP_

#include "../Longinus/common.hpp"

namespace glasssix 
{
	namespace longinus
	{
		struct Map_Val {
			face_rect_basic rect;
			int index;
			bool is_tracking;

			Map_Val() {};
			Map_Val(face_rect_basic rect, int index, bool is_tracking) : rect(rect), index(index), is_tracking(is_tracking) {};
		}; // end-struct

		struct Match_Retval {
			face_rect_basic rect;
			std::string id;
			bool is_new;

			Match_Retval() {};
			Match_Retval(face_rect_basic rect, std::string id, bool is_new) : rect(rect), id(id), is_new(is_new) {};
		}; // end-struct
	}
}
#endif // !_MATCH_RECTOR_HPP_
