#ifndef _NPCALCULATOR_HPP_
#define _NPCALCULATOR_HPP_

#include "../Excalibur/tensor.hpp"
#include <memory>

namespace glasssix
{
	class npcalculator
	{
	public:
		int objSize;
		int numStages;
		int numBranchNodes;
		int numLeafNodes;
		int numScales;
		float scaleFactor;
		//
		const char* modelversion;
		//
		std::shared_ptr<excalibur::tensor<int> > winSize;
		std::shared_ptr<excalibur::tensor<int> > pixelx;
		std::shared_ptr<excalibur::tensor<int> > pixely;
		std::shared_ptr<excalibur::tensor<int> > treeRoot;
		//npd table
		std::shared_ptr<excalibur::tensor<unsigned char> > fea;
		std::shared_ptr<excalibur::tensor<unsigned char> > cutpoint;
		std::shared_ptr<excalibur::tensor<int> > leftChild;
		std::shared_ptr<excalibur::tensor<int> > rightChild;
		std::shared_ptr<excalibur::tensor<float> > stageThreshold;
		std::shared_ptr<excalibur::tensor<float> > fit;

	private:
		void init();
		void prepare(int os, int nst, int nb, int nl, float sf, int nsa);
		void load();
		void load(const char* modelpath);
#ifdef HARDCODE_REQUIRE
	public:
		void hardcode2hpp(const char* hpppath);
	private:
		void writedata(const float* data, int len);
		void writedata(const int* data, int len);
		void writedata(const unsigned char* data, int len);
		std::ofstream out;
		int pos;
#endif

		int device_;
	public:
		npcalculator();
		npcalculator(int device = -1);
		npcalculator(const char* modelpath, int device = -1);
		~npcalculator(){};
	};
}

#endif
