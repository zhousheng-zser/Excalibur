#ifndef HARDCODE_HPP
#define HARDCODE_HPP

namespace glasssix
{
	namespace longinus
	{
		typedef struct HardCodeWeak
		{
			const int *feaInfo;
			int featureIndex;
			double weakThreshold;
			const double *regression_value;
		}HardCodeWeak;

		typedef struct HardCodeRomanciaCascade
		{
			int win_width;
			int win_height;
			int face_width;
			int face_height;
			int numStages;
			int numWeaks;
			int lbp_mode;
			int fea_mode;
			const int *weaksPerStage;
			const double *stageThresholdPerStage;
			const double *falseAlarmPerStage;
			const struct HardCodeWeak *p_weaks;
		}HardCodeCascade;
	}
}

#endif