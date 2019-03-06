#include "common.hpp"
#include <algorithm>

using namespace glasssix::longinus;

inline int is_equal_rect(const FaceRect &r1, const FaceRect &r2)
{
	int delta10x = std::min(r1.width, r2.width) + std::min(r1.height, r2.height);
	return abs(r1.x - r2.x) * 10 <= delta10x &&
		abs(r1.y - r2.y) * 10 <= delta10x &&
		abs(r1.x + r1.width - r2.x - r2.width) * 10 <= delta10x &&
		abs(r1.y + r1.height - r2.y - r2.height) * 10 <= delta10x;
}

inline int intersectionArea(FaceRect &r1, FaceRect &r2)
{
	FaceRect r;
	r.x = std::max(r1.x, r2.x);
	r.y = std::max(r1.y, r2.y);
	r.width = std::min(r1.x + r1.width, r2.x + r2.width) - r.x;
	r.height = std::min(r1.y + r1.height, r2.y + r2.height) - r.y;

	if (r.width > 0 && r.height > 0)
		return r.width * r.height;
	else
		return 0;
}

// pFaces will be modified in the function
void glasssix::longinus::GroupRects(std::vector<FaceRect> &pFaces, int min_neighbors)
{
	if (min_neighbors <= 0 || pFaces.size() == 0)
		return;

	std::vector<int> nLabels(pFaces.size());
	//init label
	for (int i = 0; i < pFaces.size(); i++)
	{
		nLabels[i] = i;
	}

	// group rectangles
	// the computational cost is a little higher,
	// but it can save memory, and avoid to use a list structure.
	for (int i = 0; i < pFaces.size() - 1; i++)
	{
		for (int j = i + 1; j < pFaces.size(); j++)
		{
			if (is_equal_rect(pFaces[i], pFaces[j]))
			{
				int min_label = std::min(nLabels[i], nLabels[j]);
				int max_label = std::max(nLabels[i], nLabels[j]);

				for (int k = 0; k < pFaces.size(); k++)
					if (nLabels[k] == max_label)
						nLabels[k] = min_label;
			}
		}
	}

	std::vector<FaceRect> pFacesBuf(pFaces.size(), FaceRect());
	for (int i = 0; i < pFacesBuf.size(); i++)
	{
		int label = nLabels[i];
		pFacesBuf[label].x += pFaces[i].x;
		pFacesBuf[label].y += pFaces[i].y;
		pFacesBuf[label].width += pFaces[i].width;
		pFacesBuf[label].height += pFaces[i].height;
		pFacesBuf[label].neighbors++;
		pFacesBuf[label].confidence += pFaces[i].confidence;
	}

	int new_label = 0;
	for (int i = 0; i < pFacesBuf.size(); i++)
	{
		if (pFacesBuf[i].neighbors > 0)
		{
			int n = pFacesBuf[i].neighbors;
			pFaces[new_label].x = (pFacesBuf[i].x * 2 + n) / (2 * n);
			pFaces[new_label].y = (pFacesBuf[i].y * 2 + n) / (2 * n);
			pFaces[new_label].width = (pFacesBuf[i].width * 2 + n) / (2 * n);
			pFaces[new_label].height = (pFacesBuf[i].height * 2 + n) / (2 * n);
			pFaces[new_label].neighbors = n;

			new_label++;
		}
	}
	pFaces.resize(new_label);

	//swap
	//pFacesBuf.resize(new_label);
	//for (int i = 0; i < pFacesBuf.size(); i++)
	//{
	//	pFacesBuf[i].x = pFaces[i].x;
	//	pFacesBuf[i].y = pFaces[i].y;
	//	pFacesBuf[i].width = pFaces[i].width;
	//	pFacesBuf[i].height = pFaces[i].height;
	//	pFacesBuf[i].neighbors = pFaces[i].neighbors;
	//	pFacesBuf[i].angle = pFaces[i].angle;
	//}

	pFacesBuf.swap(pFaces);

	pFaces.resize(0);
	// filter out small face rectangles inside large face rectangles
	for (int i = 0; i < pFacesBuf.size(); i++)
	{
		bool frb1_is_good = true;
		std::vector<FaceRect>::iterator frb1 = pFacesBuf.begin() + i;
		for (int j = 0; j < pFacesBuf.size(); j++)
		{
			std::vector<FaceRect>::iterator frb2 = pFacesBuf.begin() + j;

			int area = intersectionArea(*frb1, *frb2);
			bool overlap = (area * 2 >= frb1->width * frb1->height) || (area * 2 >= frb2->width * frb2->height);

			if ((i != j) && overlap &&
				((frb1->neighbors < frb2->neighbors) ||
				((frb1->neighbors == frb2->neighbors) && (i < j))))
			{
				frb1_is_good = false;
				break;
			}
		}

		if (frb1_is_good && pFacesBuf[i].neighbors >= min_neighbors)
		{
			pFaces.push_back(pFacesBuf[i]);
		}
	}
}
