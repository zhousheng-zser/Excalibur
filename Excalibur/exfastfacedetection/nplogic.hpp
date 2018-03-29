#ifndef _NPLOGIC_HPP_
#define _NPLOGIC_HPP_
#include "npcalculator.hpp"
#include <helper_cuda.h>
namespace glasssix
{
	class nplogic
	{
		std::shared_ptr<npcalculator> model_;
		int device_;
		// Detect parameter.
		int minFace;
		int maxFace;
		float overlappingThreshold;

		// Containers for the detected faces.
		std::vector< float > xs, ys;
		std::vector< float > sizes;
		std::vector< float > scores;

		std::vector< int > Xs, Ys, Ss;
		std::vector< float > Scores;

		int numScan;
		int numDetect;

		// Temp space.
		int maxScanNum;
		int maxDetectNum;

		// Scan size.
		std::shared_ptr<tensor<char>> Tpredicate;
		std::shared_ptr<tensor<int>> Troot;
		std::shared_ptr<tensor<float>> Tlogweight;
		std::shared_ptr<tensor<int>> Tparent;
		std::shared_ptr<tensor<int>> Trank;

		// Detect size.
		std::shared_ptr<tensor<int>> Tneighbors;
		std::shared_ptr<tensor<float>> Tweight;
		std::shared_ptr<tensor<float>> Txs;
		std::shared_ptr<tensor<float>> Tys;
		std::shared_ptr<tensor<float>> Tss;

		void init(int minFace = 20, int maxFace = 400);
		void mallocsacnspace(int s);
		void mallocdetectspace(int n);
		/*void freesacnspace();
		void freedetectspace();*/
		void reset();
		float logistic(float s)
		{
			return log(1 + exp(double(s)));
		}

		int findRoot(const int* parent, int i)
		{
			if (parent[i] != i)
				return findRoot(parent, parent[i]);
			else
				return i;
		}

		int nms();
		int partition(const char* predicate, int* root);
		int floodScoreMat(std::shared_ptr<tensor<float>> mat, int mat_height, int mat_width, int rowMax, int colMax, int winStep);
		int scan_cpu(const unsigned char* I, int width, int height, int min_size);
#ifdef USE_CUDA
		//const int gridDims[10] = { 40, 80, 120, 160, 200, 240, 280, 320, 360, 400 };
		int cuda_level;
		
		int GetDeviceCUDACoreNum(int dev)
		{
			cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);
			return _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
		}
		int scan_gpu(const unsigned char* I, int width, int height, int min_size);

		template<int gridDimension>
		int scan_gpu_template(const unsigned char* I, int width, int height, int min_size);
#endif
	public:
		nplogic(int device = -1);
		nplogic(int minFace, int maxFace, int device = -1);
		void load();
		void load(const char* modelpath);
		int detect(const unsigned char* I, int width, int height, int min_size);
		~nplogic(){};

		std::vector< int >& getXs() { return Xs; }
		std::vector< int >& getYs() { return Ys; }
		std::vector< int >& getSs() { return Ss; }
		std::vector< float >& getScores() { return Scores; }
	};
}

#endif