#include "matcher.hpp"
#include <sstream>

namespace glasssix
{
	namespace longinus
	{
		// constructor
		Matcher::Matcher()
		{
		}

		// destructor
		Matcher::~Matcher()
		{
		}

		// get_guid_string: 
		// generate a random string id for each face rect
		std::string get_guid_string() 
		{
			const std::string CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
			std::string uuid = std::string(36, ' ');
			int rnd = 0;
			int r = 0;

			uuid[8] = '-';
			uuid[13] = '-';
			uuid[18] = '-';
			uuid[23] = '-';

			uuid[14] = '4';

			for (int i = 0; i < 36; i++) {
				if (i != 8 && i != 13 && i != 18 && i != 14 && i != 23) {
					if (rnd <= 0x02) {
						rnd = 0x2000000 + (std::rand() * 0x1000000) | 0;
					}
					rnd >>= 4;
					uuid[i] = CHARS[(i == 19) ? ((rnd & 0xf) & 0x3) | 0x8 : rnd & 0xf];
				}
			}
			return uuid;
		} // end-guid-string


		  // assign_id: 
		  // generate a unique face id for every face rect
		  // params:
		  // - output: stores the face rects and their corresponding id
		  // - rects: a vector of face rects
		void assign_id(std::vector<Match_Retval> &output, const std::vector<FaceRect> &rects) {

			for (size_t i = 0; i < rects.size(); i++) {
				Match_Retval t(rects[i], get_guid_string(), true);
				output.push_back(t);
			}
		} // end-detect


		  // cleanse_map: 
		  // remove redundant elements from the map;
		  // if an element has existed for more than #num frames, delete it;
		  // if an element is close to the border, delete it;
		  // params:
		  // - width: the width of each frame
		  // - height: the height of each frame
		  // - curr_index: the index of current frame
		  // - interval: interval between 2 consecutive frames
		void cleanse_map(std::map<std::string, Map_Val> &matcher_map, const int curr_index, const int interval) {
			int FRAME_INTERVAL = interval; // the number of frames that an element in the map can exist

										   // go through all elements in the map
			std::map<std::string, Map_Val>::iterator it;
			it = matcher_map.begin();
			while (it != matcher_map.end())
			{
				int start_index = it->second.index;
				FaceRect temp_rect = it->second.rect;

				if (matcher_map.size() > 1 && (abs(curr_index - start_index) > FRAME_INTERVAL)) {

					matcher_map.erase(it++);

				}
				else {
					it->second.is_tracking = false; // check if the current map element is being tracking
					it++;
				} // end-if-else
			} // end-while
		} // end-cleanse-map


		  // check_distance: 
		  // compute the distances between current element and other elements in the map,
		  // and choose the smallest one as its predecessor, taking predecessor's id as current element's id;
		  // params: 
		  // - dense_: how many people are in the frame, if the number's greather than 3, decrease the distance constraint between faces;
		  //			 if it's not greater than 3, increase the distance constraint between faces.
		  // - rect: current face rect
		std::string check_distance(const Match_Retval &rect, std::map<std::string, Map_Val> &matcher_map, const float dense_) {
			FaceRect temp_rect = rect.rect; // fetch face rect 
			std::map<std::string, Map_Val>::iterator it; // go through all the elements in the map
			it = matcher_map.begin();
			double smaller = DBL_MAX; // max value of double precision
			std::string retval = "";
			while (it != matcher_map.end())
			{
				FaceRect temp_matcher = it->second.rect;

				// if the distance between the rect and any matcher is greater than a threshold, return the matcher's string id
				Point rect_center = Point(temp_rect.x + temp_rect.width / 2, temp_rect.y + temp_rect.height / 2);

				Point matcher_center = Point(temp_matcher.x + temp_matcher.width / 2, temp_matcher.y + temp_matcher.height / 2);

				double dist = sqrt((rect_center.x - matcher_center.x) * (rect_center.x - matcher_center.x) +
					(rect_center.y - matcher_center.y) * (rect_center.y - matcher_center.y));

				if (dist < smaller && dist <= temp_rect.width * dense_ && (!it->second.is_tracking)) {

					smaller = dist;

					retval = it->first;
				} // end-if
				it++;
			} // end-while
			return retval;
		} // end-check-distance


		  // match:
		  // In each frame, match the face rects.
		  // params:
		  // - cols: width of each frame
		  // - rows: height of each frame
		  // - faceRect: a set of face rects that were detected via camera
		  // - resize_factor: to resize each frame
		  // - frame_extract_frequency: how frequently to extract one frame
		  // - i: the current frame index
		std::vector<Match_Retval> Matcher::match(std::vector<FaceRect> &faceRect,
			const int frame_extract_frequency) {
			// resize the faceRect to match the new size of frame
			for (size_t j = 0; j < faceRect.size(); j++)
			{
				faceRect[j].x = (int)(faceRect[j].x * 0.4);
				faceRect[j].y = (int)(faceRect[j].y * 0.4);
				faceRect[j].width = (int)(faceRect[j].width * 0.4);
				faceRect[j].height = (int)(faceRect[j].height * 0.4);
			} // end-for-j

			index++; // count how many frames matched
			int current_frame_index = (index - 1) * frame_extract_frequency; // current frame index

			if (index % 200000 == 0) {
				index = 0;
			} // end-if

			std::vector<Match_Retval> output;

			// only count face rects every frame_extract_frequency frames, e.g. 10 frames 
			if (current_frame_index % (10 * frame_extract_frequency) == 0) {
				num_face_rect = 0;
			} // end-if 

			  // assign a face id to each detected face rect
			assign_id(output, faceRect);

			num_face_rect += output.size(); // add current num of face rects to num_face_rect 

											// compute the average num of face rects in last 10 frames
			float avg_face_rect = num_face_rect / ((float)(current_frame_index % (10 * frame_extract_frequency)) / frame_extract_frequency);

			// remove all elements that have existed for more than num frames
			// if more than 2 faces appear in one frame, decrease the frame interval of each face;
			// otherwise, increase the frame interval.
			if (avg_face_rect > 2) {
				cleanse_map(matcher_map, current_frame_index, 20);
			}
			else {
				cleanse_map(matcher_map, current_frame_index, 200);
			} // end-if-else

			for (size_t j = 0; j < output.size(); j++)
			{

				// add the info of each matcher into the map
				std::string str_id = "";
				if (!matcher_map.empty()) {

					// find the distance between current rect and every matcher in the matcher_map
					std::string map_str_id = "";

					// if more than 3 faces appear in current frame, increase the distance constraint between faces;
					// otherwise, decrease the distance requirement.
					if (output.size() > 3)
						map_str_id = check_distance(output[j], matcher_map, 0.2f * frame_extract_frequency); // check if current rect and any rect are close in the map
					else
						map_str_id = check_distance(output[j], matcher_map, 0.5f * frame_extract_frequency); // check if current rect and any rect are close in the map

																											 // if current rect is very close to a rect in the map, update matcher in the map
					if (map_str_id != "") {
						str_id = map_str_id;
						// update the matcher in the map
						output[j].is_new = false;
						Map_Val mv(output[j].rect, current_frame_index, true);
						matcher_map[str_id] = mv;

					}
					else {
						// if map does not contain that element, insert it into map
						output[j].is_new = true;

						Map_Val mv(output[j].rect, current_frame_index, true);

						str_id = output[j].id;
						std::pair<std::string, Map_Val> pair_t = make_pair(str_id, mv);
						matcher_map.insert(pair_t);
					} // end-if-else
				}
				else {
					// matcher_map is empty, so insert element directly
					output[j].is_new = true;

					Map_Val mv(output[j].rect, current_frame_index, true); // insert the current rect

					str_id = output[j].id;
					std::pair<std::string, Map_Val> pair_t = make_pair(str_id, mv);
					matcher_map.insert(pair_t);
				} // end-if-else

				output[j].id = str_id;

			} // end-for-j	

			return output;
		} // end-match

	}
} // end-namespace
