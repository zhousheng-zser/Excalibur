#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <string>
#include <memory> 

#ifdef EXPORT_LONGINUS
#undef EXPORT_LONGINUS
#ifdef _MSC_VER // For Windows
#ifdef _WINDLL // Dynamic lib
#define EXPORT_LONGINUS __declspec(dllexport)
#else // Static lib
#define EXPORT_LONGINUS
#endif //!_WINDLL
#elif defined(__linux__) // For Linux
#define EXPORT_LONGINUS 
#endif
#else  
#ifdef _MSC_VER
#define EXPORT_LONGINUS __declspec(dllimport)
#elif defined(__linux__)
#define EXPORT_LONGINUS 
#endif
#endif

namespace glasssix
{
	namespace longinus
	{
		typedef struct face_rect_basic {
			int x;
			int y;
			int width;
			int height;
			int neighbors;
			float confidence;

			face_rect_basic() :x(0), y(0), width(0), height(0), neighbors(0), confidence(0.0) {}
			face_rect_basic(int x_, int y_, int width_, int height_, int neighbors_, float confidence_)
				:x(x_), y(y_), width(width_), height(height_), neighbors(neighbors_), confidence(confidence_) {}

			bool operator <(const face_rect_basic& s) const
			{
				return (width * height) > (s.width * s.height);
			}
		} face_rect_basic;

		class BaseLonginusCascade;
		class EXPORT_LONGINUS LonginusDetector;

		typedef struct CandidateRect : public face_rect_basic
		{
			int index_in_image_pyramids;
			int ix;
			int iy;
			int xstep;
			int ystep;
			int xmax;
			int ymax;
			std::shared_ptr<BaseLonginusCascade> cascade;
			CandidateRect() :index_in_image_pyramids(-1), ix(-1), iy(-1), xstep(0), ystep(0), xmax(-1), ymax(-1), cascade(nullptr) {}
			CandidateRect(int x_, int y_, int width_, int height_, int neighbors_, float confidence_, 
				int index_in_image_pyramids_, int ix_, int iy_, int xstep_, int ystep_, int xmax_, int ymax_, std::shared_ptr<BaseLonginusCascade> cascade_)
			:face_rect_basic(x_, y_, width_, height_, neighbors_, confidence_), index_in_image_pyramids(-1), ix(-1), iy(-1), xstep(0), ystep(0), xmax(-1), ymax(-1), cascade(cascade_) {}
		}CandidateRect;

		void GroupRects(std::vector<face_rect_basic> &pFaces, int min_neighbors);

		typedef struct Point2i
		{
			int x;
			int y;
			Point2i() :x(-1), y(-1){}
			Point2i(int ix, int iy) :x(ix), y(iy) {}
		}Point;

		typedef struct Point2f
		{
			float x;
			float y;
			Point2f() :x(-1), y(-1) {}
			Point2f(float ix, float iy) :x(ix), y(iy) {}
		}Point2f;

		static Point2f operator+(const Point2f &lhs, const Point2f &rhs) {
			Point2f result;
			result.x = lhs.x + rhs.x;
			result.y = lhs.y + rhs.y;

			return result;
		}

		static Point2f operator-(const Point2f &lhs, const Point2f &rhs) {
			Point2f result;
			result.x = lhs.x - rhs.x;
			result.y = lhs.y - rhs.y;

			return result;
		}

		static Point2f operator/(const Point2f &lhs, float rhs) {
			Point2f result;
			result.x = lhs.x / rhs;
			result.y = lhs.y / rhs;

			return result;
		}

		static Point2f operator*(const Point2f &lhs, float rhs) {
			Point2f result;
			result.x = lhs.x * rhs;
			result.y = lhs.y * rhs;

			return result;
		}

		static float operator^(const Point2f &lhs, const Point2f &rhs) {
			auto dx = lhs.x - rhs.x;
			auto dy = lhs.y - rhs.y;
			return std::sqrt(dx * dx + dy * dy);
		}

		/**
		 * line for ax + by + c = 0
		 */
		class Line 
		{
		public:
			Line() = default;
			Line(float a, float b, float c)
				: a(a), b(b), c(c) {}

			Line(const Point2f &a, const Point2f &b) 
			{
				auto x1 = a.x;
				auto y1 = a.y;
				auto x2 = b.x;
				auto y2 = b.y;
				// for (y2-y1)x-(x2-x1)y-x1(y2-y1)+y1(x2-x1)=0
				this->a = y2 - y1;
				this->b = x1 - x2;
				this->c = y1 * (x2 - x1) - x1 * (y2 - y1);
			}

			float distance(const Point2f &p) const 
			{
				return std::fabs(a * p.x + b * p.y + c) / std::sqrt(a * a + b * b);
			}

			static bool near_zero(float f) 
			{
				return f <= DBL_EPSILON && -f <= DBL_EPSILON;
			}

			Point2f projection(const Point2f &p) const 
			{
				if (near_zero(a)) 
				{
					Point2f result;
					result.x = p.x;
					result.y = -c / b;
					return  result;
				}
				if (near_zero(b)) 
				{
					Point2f result;
					result.x = -c / a;
					result.y = p.y;
					return result;
				}
				// y = kx + b  <==>  ax + by + c = 0
				auto k = -a / b;
				Point2f o = { 0, -c / b };
				Point2f project;
				project.x = (float)((p.x / k + p.y - o.y) / (1 / k + k));
				project.y = (float)(-1 / k * (project.x - p.x) + p.y);
				return project;
			}

			float a = 0;
			float b = 0;
			float c = 0;
		};

		typedef struct ScaledMatrix
		{
			int factor1024x;
			int winStep;
			ScaledMatrix() :factor1024x(0), winStep(0) {}
			ScaledMatrix(int factor1024x_, int winStep_) :
				factor1024x(factor1024x_), winStep(winStep_){}
		}ScaledMatrix;

		typedef struct FaceBox {
			float xmin;
			float ymin;
			float xmax;
			float ymax;
			float score;
		} FaceBox;

		typedef struct FaceInfomation {
			float bbox_reg[4];
			float landmark[10];
			float headpose[3];
			FaceBox bbox;
		} FaceInfomation;

		// Use 5 landmarks to estimate head pose
		inline void evaluate_pose(const Point2f* points, float &yaw, float &pitch, float &roll)
		{
			static const float nose_center = 0.5f;
			// static const float roll0 = 1 / 6.0f;
			// static const float yaw0 = 0.2f;
			// static const float pitch0 = 0.2f;

			const auto point_center_eye = (points[0] + points[1]) / 2;
			const auto point_center_mouth = (points[3] + points[4]) / 2;

			Line line_eye_mouth(point_center_eye, point_center_mouth);

			const auto vector_left2right = points[1] - points[0];

			const auto rad = atan2(vector_left2right.y, vector_left2right.x);
			const auto angle = rad * 180 * 3.1415926;

			const auto roll_dist = fabs(angle) / 180;

			const auto raw_yaw_dist = line_eye_mouth.distance(points[2]);
			const auto yaw_dist = raw_yaw_dist / (points[0] ^ points[1]);

			const auto point_suppose_projection = point_center_eye * nose_center + point_center_mouth * (1 - nose_center);
			const auto point_projection = line_eye_mouth.projection(points[2]);
			const auto raw_pitch_dist = point_projection ^ point_suppose_projection;
			const auto pitch_dist = raw_pitch_dist / (point_center_eye ^ point_center_mouth);

			roll = roll_dist;
			yaw = yaw_dist;
			pitch = pitch_dist;
		}
	}
}

#endif