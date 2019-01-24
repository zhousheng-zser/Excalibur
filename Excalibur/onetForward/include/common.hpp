#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <string>
#include <memory> 

#ifdef ROMANCIA_LIB
#undef ROMANCIA_LIB
#ifdef _WIN32
#define ROMANCIA_LIB __declspec(dllexport)
#elif defined(linux)
#define ROMANCIA_LIB 
#endif
#else  
#ifdef _WIN32
#define ROMANCIA_LIB __declspec(dllimport)
#elif defined(linux)
#define ROMANCIA_LIB 
#endif
#endif

namespace glasssix
{
	namespace longinus
	{
		typedef struct FaceRect {
			int x;
			int y;
			int width;
			int height;
			int neighbors;
			int angle;
			double confidence;

			FaceRect() :x(0), y(0), width(0), height(0), neighbors(0), angle(0), confidence(0.0) {}
			FaceRect(int x_, int y_, int width_, int height_, int neighbors_, int angle_, double confidence_)
				:x(x_), y(y_), width(width_), height(height_), neighbors(neighbors_), angle(angle_), confidence(confidence_) {}
		} FaceRect;

		class BaseRomanciaCascade;
		class ROMANCIA_LIB RomanciaDetector;

		typedef struct CandidateRect : public FaceRect
		{
			int index_in_image_pyramids;
			int ix;
			int iy;
			int xstep;
			int ystep;
			int xmax;
			int ymax;
			std::shared_ptr<BaseRomanciaCascade> cascade;
			CandidateRect() :index_in_image_pyramids(-1), ix(-1), iy(-1), xstep(0), ystep(0), xmax(-1), ymax(-1), cascade(nullptr) {}
			CandidateRect(int x_, int y_, int width_, int height_, int neighbors_, int angle_, double confidence_, 
				int index_in_image_pyramids_, int ix_, int iy_, int xstep_, int ystep_, int xmax_, int ymax_, std::shared_ptr<BaseRomanciaCascade> cascade_)
			:FaceRect(x_, y_, width_, height_, neighbors_, angle_, confidence_), index_in_image_pyramids(-1), ix(-1), iy(-1), xstep(0), ystep(0), xmax(-1), ymax(-1), cascade(cascade_) {}
		}CandidateRect;

		void GroupRects(std::vector<FaceRect> &pFaces, int min_neighbors);

		typedef struct Point
		{
			int x;
			int y;
			Point() :x(-1), y(-1){}
			Point(int ix, int iy) :x(ix), y(iy) {}
		}Point;

		typedef struct ScaledMatrix
		{
			int factor1024x;
			int winStep;
			ScaledMatrix() :factor1024x(0), winStep(0) {}
			ScaledMatrix(int factor1024x_, int winStep_) :
				factor1024x(factor1024x_), winStep(winStep_){}
		}ScaledMatrix;
	}
}

#endif