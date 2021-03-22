#include "cpu.hpp"

#ifdef _MSC_VER //Windows
#include <Windows.h>
#else // *nix
#include "unistd.h"
#endif

#ifdef _MSC_VER
#include <intrin.h>
#elif defined(x86)
#ifdef __GNUC__
#include <x86intrin.h>
#endif
#endif
#include <vector>
#include <array>
#include <bitset>
#include <cstring>
namespace glasssix
{
#ifdef x86

	std::string cpu_vendor()
	{
		int eax[4];
		__cpuidex(eax, 0, 0);
		char vendor[0x20];
		memset(vendor, 0, sizeof(vendor));
		*reinterpret_cast<int*>(vendor) = eax[1];
		*reinterpret_cast<int*>(vendor + 4) = eax[3];
		*reinterpret_cast<int*>(vendor + 8) = eax[2];
		const std::string vendor_ = vendor;
		return vendor_;
	}

	bool isIntel()
	{
		const std::string vendor_ = cpu_vendor();
		return vendor_ == "GenuineIntel";
	}

	bool isAMD()
	{
		const std::string vendor_ = cpu_vendor();
		return vendor_ == "AMDisbetter!" || vendor_ == "AuthenticAMD";
	}

	std::string cpu_brand()
	{
		std::string brand_;
		int eax[4];
		// Calling __cpuid with 0x80000000 as the function_id argument
		// gets the number of the highest valid extended ID.
		__cpuidex(eax, 0x80000000, 0);
		const int nExIds_ = eax[0];
		char brand[0x40];
		memset(brand, 0, sizeof(brand));
		std::vector<std::array<int, 4>> extdata_;
		for (int i = 0x80000000; i <= nExIds_; ++i)
		{
			__cpuidex(eax, i, 0);
			extdata_.push_back(std::array<int, 4>{eax[0], eax[1], eax[2], eax[3]});
		}
		// Interpret CPU brand string if reported
		if (nExIds_ >= 0x80000004)
		{
			memcpy(brand, extdata_[2].data(), sizeof(eax));
			memcpy(brand + 16, extdata_[3].data(), sizeof(eax));
			memcpy(brand + 32, extdata_[4].data(), sizeof(eax));
			brand_ = brand;
		}
		else
		{
			brand_ = std::string("");
		}
		return brand_;
	}

	bool support_MMX()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_EDX_ = eax[3];
		return f_1_EDX_[23];
	}

	bool support_SSE()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_EDX_ = eax[3];
		return f_1_EDX_[25];
	}

	bool support_SSE2()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_EDX_ = eax[3];
		return f_1_EDX_[26];
	}

	bool support_SSE3()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_ECX_ = eax[2];
		return f_1_ECX_[0];
	}

	bool support_SSSE3()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_ECX_ = eax[2];
		return f_1_ECX_[9];
	}

	bool support_FMA()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_ECX_ = eax[2];
		return f_1_ECX_[12];
	}

	bool support_SSE41()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_ECX_ = eax[2];
		return f_1_ECX_[19];
	}

	bool support_SSE42()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_ECX_ = eax[2];
		return f_1_ECX_[20];
	}

	bool support_AVX()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_ECX_ = eax[2];
		return f_1_ECX_[28];
	}

	bool support_F16C()
	{
		int eax[4];
		__cpuidex(eax, 1, 0);
		std::bitset<32> f_1_ECX_ = eax[2];
		return f_1_ECX_[29];
	}

	bool support_AVX2()
	{
		int eax[4];
		__cpuidex(eax, 7, 0);
		std::bitset<32> f_7_EBX_ = eax[1];
		return f_7_EBX_[5];
	}

	bool support_AVX512F()
	{
		int eax[4];
		__cpuidex(eax, 7, 0);
		std::bitset<32> f_7_EBX_ = eax[1];
		return f_7_EBX_[16];
	}

	/// Intel?Xeon Phi only
	bool support_AVX512PF()
	{
		int eax[4];
		__cpuidex(eax, 7, 0);
		std::bitset<32> f_7_EBX_ = eax[1];
		return f_7_EBX_[26];
	}

	/// Intel?Xeon Phi only
	bool support_AVX512ER()
	{
		int eax[4];
		__cpuidex(eax, 7, 0);
		std::bitset<32> f_7_EBX_ = eax[1];
		return f_7_EBX_[27];
	}

	bool support_AVX512CD()
	{
		int eax[4];
		__cpuidex(eax, 7, 0);
		std::bitset<32> f_7_EBX_ = eax[1];
		return f_7_EBX_[28];
	}

	void queryCacheSizes_intel_direct(int& l1, int& l2, int& l3)
	{
		int abcd[4];
		l1 = l2 = l3 = 0;
		int cache_id = 0;
		int cache_type = 0;
		do
		{
			abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
			__cpuidex(abcd, 0x4, cache_id);
			cache_type = (abcd[0] & 0x0F) >> 0;
			if (cache_type == 1 || cache_type == 3) // data or unified cache
			{
				const int cache_level = (abcd[0] & 0xE0) >> 5;  // A[7:5]
				const int ways = (abcd[1] & 0xFFC00000) >> 22; // B[31:22]
				const int partitions = (abcd[1] & 0x003FF000) >> 12; // B[21:12]
				const int line_size = (abcd[1] & 0x00000FFF) >> 0; // B[11:0]
				const int sets = (abcd[2]);                    // C[31:0]

				const int cache_size = (ways + 1) * (partitions + 1) * (line_size + 1) * (sets + 1);

				switch (cache_level)
				{
				case 1: l1 = cache_size; break;
				case 2: l2 = cache_size; break;
				case 3: l3 = cache_size; break;
				default: break;
				}
			}
			cache_id++;
		} while (cache_type > 0 && cache_id < 16);
	}

	void queryCacheSizes_intel_codes(int& l1, int& l2, int& l3)
	{
		int abcd[4];
		abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
		l1 = l2 = l3 = 0;
		__cpuidex(abcd, 0x00000002, 0);
		unsigned char * bytes = reinterpret_cast<unsigned char *>(abcd) + 2;
		bool check_for_p2_core2 = false;
		for (int i = 0; i < 14; ++i)
		{
			switch (bytes[i])
			{
			case 0x0A: l1 = 8; break;   // 0Ah   data L1 cache, 8 KB, 2 ways, 32 byte lines
			case 0x0C: l1 = 16; break;  // 0Ch   data L1 cache, 16 KB, 4 ways, 32 byte lines
			case 0x0E: l1 = 24; break;  // 0Eh   data L1 cache, 24 KB, 6 ways, 64 byte lines
			case 0x10: l1 = 16; break;  // 10h   data L1 cache, 16 KB, 4 ways, 32 byte lines (IA-64)
			case 0x15: l1 = 16; break;  // 15h   code L1 cache, 16 KB, 4 ways, 32 byte lines (IA-64)
			case 0x2C: l1 = 32; break;  // 2Ch   data L1 cache, 32 KB, 8 ways, 64 byte lines
			case 0x30: l1 = 32; break;  // 30h   code L1 cache, 32 KB, 8 ways, 64 byte lines
			case 0x60: l1 = 16; break;  // 60h   data L1 cache, 16 KB, 8 ways, 64 byte lines, sectored
			case 0x66: l1 = 8; break;   // 66h   data L1 cache, 8 KB, 4 ways, 64 byte lines, sectored
			case 0x67: l1 = 16; break;  // 67h   data L1 cache, 16 KB, 4 ways, 64 byte lines, sectored
			case 0x68: l1 = 32; break;  // 68h   data L1 cache, 32 KB, 4 ways, 64 byte lines, sectored
			case 0x1A: l2 = 96; break;   // code and data L2 cache, 96 KB, 6 ways, 64 byte lines (IA-64)
			case 0x22: l3 = 512; break;   // code and data L3 cache, 512 KB, 4 ways (!), 64 byte lines, dual-sectored
			case 0x23: l3 = 1024; break;   // code and data L3 cache, 1024 KB, 8 ways, 64 byte lines, dual-sectored
			case 0x25: l3 = 2048; break;   // code and data L3 cache, 2048 KB, 8 ways, 64 byte lines, dual-sectored
			case 0x29: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 8 ways, 64 byte lines, dual-sectored
			case 0x39: l2 = 128; break;   // code and data L2 cache, 128 KB, 4 ways, 64 byte lines, sectored
			case 0x3A: l2 = 192; break;   // code and data L2 cache, 192 KB, 6 ways, 64 byte lines, sectored
			case 0x3B: l2 = 128; break;   // code and data L2 cache, 128 KB, 2 ways, 64 byte lines, sectored
			case 0x3C: l2 = 256; break;   // code and data L2 cache, 256 KB, 4 ways, 64 byte lines, sectored
			case 0x3D: l2 = 384; break;   // code and data L2 cache, 384 KB, 6 ways, 64 byte lines, sectored
			case 0x3E: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 64 byte lines, sectored
			case 0x40: l2 = 0; break;   // no integrated L2 cache (P6 core) or L3 cache (P4 core)
			case 0x41: l2 = 128; break;   // code and data L2 cache, 128 KB, 4 ways, 32 byte lines
			case 0x42: l2 = 256; break;   // code and data L2 cache, 256 KB, 4 ways, 32 byte lines
			case 0x43: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 32 byte lines
			case 0x44: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 4 ways, 32 byte lines
			case 0x45: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 4 ways, 32 byte lines
			case 0x46: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 4 ways, 64 byte lines
			case 0x47: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 8 ways, 64 byte lines
			case 0x48: l2 = 3072; break;   // code and data L2 cache, 3072 KB, 12 ways, 64 byte lines
			case 0x49: if (l2 != 0) l3 = 4096; else { check_for_p2_core2 = true; l3 = l2 = 4096; } break;// code and data L3 cache, 4096 KB, 16 ways, 64 byte lines (P4) or L2 for core2
			case 0x4A: l3 = 6144; break;   // code and data L3 cache, 6144 KB, 12 ways, 64 byte lines
			case 0x4B: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 16 ways, 64 byte lines
			case 0x4C: l3 = 12288; break;   // code and data L3 cache, 12288 KB, 12 ways, 64 byte lines
			case 0x4D: l3 = 16384; break;   // code and data L3 cache, 16384 KB, 16 ways, 64 byte lines
			case 0x4E: l2 = 6144; break;   // code and data L2 cache, 6144 KB, 24 ways, 64 byte lines
			case 0x78: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 4 ways, 64 byte lines
			case 0x79: l2 = 128; break;   // code and data L2 cache, 128 KB, 8 ways, 64 byte lines, dual-sectored
			case 0x7A: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 64 byte lines, dual-sectored
			case 0x7B: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 64 byte lines, dual-sectored
			case 0x7C: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 64 byte lines, dual-sectored
			case 0x7D: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 8 ways, 64 byte lines
			case 0x7E: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 128 byte lines, sect. (IA-64)
			case 0x7F: l2 = 512; break;   // code and data L2 cache, 512 KB, 2 ways, 64 byte lines
			case 0x80: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 64 byte lines
			case 0x81: l2 = 128; break;   // code and data L2 cache, 128 KB, 8 ways, 32 byte lines
			case 0x82: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 32 byte lines
			case 0x83: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 32 byte lines
			case 0x84: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 32 byte lines
			case 0x85: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 8 ways, 32 byte lines
			case 0x86: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 64 byte lines
			case 0x87: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 64 byte lines
			case 0x88: l3 = 2048; break;   // code and data L3 cache, 2048 KB, 4 ways, 64 byte lines (IA-64)
			case 0x89: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 4 ways, 64 byte lines (IA-64)
			case 0x8A: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 4 ways, 64 byte lines (IA-64)
			case 0x8D: l3 = 3072; break;   // code and data L3 cache, 3072 KB, 12 ways, 128 byte lines (IA-64)

			default: break;
			}
		}
		if (check_for_p2_core2 && l2 == l3)
			l3 = 0;
		l1 *= 1024;
		l2 *= 1024;
		l3 *= 1024;
	}

	void queryCacheSizes_amd(int& l1, int& l2, int& l3)
	{
		int abcd[4];
		abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
		__cpuidex(abcd, 0x80000005, 0);
		l1 = (abcd[2] >> 24) * 1024; // C[31:24] = L1 size in KB
		abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
		__cpuidex(abcd, 0x80000006, 0);
		l2 = (abcd[2] >> 16) * 1024; // C[31;16] = l2 cache size in KB
		l3 = ((abcd[3] & 0xFFFC000) >> 18) * 512 * 1024; // D[31;18] = l3 cache size in 512KB
	}

	void queryCacheSizes_intel(int& l1, int& l2, int& l3, int max_std_funcs)
	{
		if (max_std_funcs >= 4)
			queryCacheSizes_intel_direct(l1, l2, l3);
		else
			queryCacheSizes_intel_codes(l1, l2, l3);
	}

	/// For L1	cache, it returns size for each logical cores(Hyper Threading Technology) in Bytes;\n
	/// For L2	cache, it returns size for each physical cores in Bytes;\n
	/// For L3	cache, it returns size for the whole CPU in Bytes;\n
	void queryCacheSizes(int& l1, int& l2, int& l3)
	{
		int abcd[4];

		// identify the CPU vendor
		__cpuidex(abcd, 0x0, 0);
		const int max_std_funcs = abcd[1];
		if (isIntel())
		{
			queryCacheSizes_intel(l1, l2, l3, max_std_funcs);
		}
		else if (isAMD())
		{
			queryCacheSizes_amd(l1, l2, l3);
		}
		else
		{
			// Un-supportted CPU vendor
			l1 = l2 = l3 = -1;
		}
	}

	unsigned int queryLogicalProcessors()
	{
#ifdef _MSC_VER
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		return sysInfo.dwNumberOfProcessors;
#else
		return sysconf(_SC_NPROCESSORS_CONF);
#endif
	}
#endif //!x86

#ifdef arm
#ifdef __ANDROID__

	// extract the ELF HW capabilities bitmap from /proc/self/auxv
	unsigned int get_elf_hwcap_from_proc_self_auxv()
	{
		FILE* fp = fopen("/proc/self/auxv", "rb");
		if (!fp)
		{
			return 0;
		}

#define AT_HWCAP 16
#define AT_HWCAP2 26
#if __aarch64__

		struct { uint64_t tag; uint64_t value; } entry;
#else
		struct { unsigned int tag; unsigned int value; } entry;

#endif

		unsigned int result = 0;
		while (!feof(fp))
		{
			int nread = fread((char*)&entry, sizeof(entry), 1, fp);
			if (nread != 1)
				break;

			if (entry.tag == 0 && entry.value == 0)
				break;

			if (entry.tag == AT_HWCAP)
			{
				result = entry.value;
				break;
			}
		}

		fclose(fp);

		return result;
	}

	unsigned int g_hwcaps = get_elf_hwcap_from_proc_self_auxv();

#if __aarch64__
	// from arch/arm64/include/uapi/asm/hwcap.h
#define HWCAP_ASIMD     (1 << 1)
#define HWCAP_ASIMDHP   (1 << 10)
#else
	// from arch/arm/include/uapi/asm/hwcap.h
#define HWCAP_NEON      (1 << 12)
#define HWCAP_VFPv4     (1 << 16)
#endif

#endif // __ANDROID__

#if __IOS__
	unsigned int get_hw_cpufamily()
	{
		unsigned int value = 0;
		size_t len = sizeof(value);
		sysctlbyname("hw.cpufamily", &value, &len, NULL, 0);
		return value;
	}

	cpu_type_t get_hw_cputype()
	{
		cpu_type_t value = 0;
		size_t len = sizeof(value);
		sysctlbyname("hw.cputype", &value, &len, NULL, 0);
		return value;
	}

	cpu_subtype_t get_hw_cpusubtype()
	{
		cpu_subtype_t value = 0;
		size_t len = sizeof(value);
		sysctlbyname("hw.cpusubtype", &value, &len, NULL, 0);
		return value;
	}

	unsigned int g_hw_cpufamily = get_hw_cpufamily();
	cpu_type_t g_hw_cputype = get_hw_cputype();
	cpu_subtype_t g_hw_cpusubtype = get_hw_cpusubtype();
#endif // __IOS__

	int cpu_support_arm_neon()
	{
#ifdef __ANDROID__
#if __aarch64__
		return g_hwcaps & HWCAP_ASIMD;
#else
		return g_hwcaps & HWCAP_NEON;
#endif
#elif __IOS__
#if __aarch64__
		return g_hw_cputype == CPU_TYPE_ARM64;
#else
		return g_hw_cputype == CPU_TYPE_ARM && g_hw_cpusubtype > CPU_SUBTYPE_ARM_V7;
#endif
#else
		return 0;
#endif
	}

	int cpu_support_arm_vfpv4()
	{
#ifdef __ANDROID__
#if __aarch64__
		// neon always enable fma and fp16
		return g_hwcaps & HWCAP_ASIMD;
#else
		return g_hwcaps & HWCAP_VFPv4;
#endif
#elif __IOS__
#if __aarch64__
		return g_hw_cputype == CPU_TYPE_ARM64;
#else
		return g_hw_cputype == CPU_TYPE_ARM && g_hw_cpusubtype > CPU_SUBTYPE_ARM_V7S;
#endif
#else
		return 0;
#endif
	}

	int cpu_support_arm_asimdhp()
	{
#ifdef __ANDROID__
#if __aarch64__
		return g_hwcaps & HWCAP_ASIMDHP;
#else
		return 0;
#endif
#elif __IOS__
#if __aarch64__
#ifndef CPUFAMILY_ARM_HURRICANE
#define CPUFAMILY_ARM_HURRICANE 0x67ceee93
#endif
#ifndef CPUFAMILY_ARM_MONSOON_MISTRAL
#define CPUFAMILY_ARM_MONSOON_MISTRAL 0xe81e7ef6
#endif
		return g_hw_cpufamily == CPUFAMILY_ARM_HURRICANE || g_hw_cpufamily == CPUFAMILY_ARM_MONSOON_MISTRAL;
#else
		return 0;
#endif
#else
		return 0;
#endif
	}

	int get_cpucount()
	{
#ifdef __ANDROID__
		// get cpu count from /proc/cpuinfo
		FILE* fp = fopen("/proc/cpuinfo", "rb");
		if (!fp)
			return 1;

		int count = 0;
		char line[1024];
		while (!feof(fp))
		{
			char* s = fgets(line, 1024, fp);
			if (!s)
				break;

			if (memcmp(line, "processor", 9) == 0)
			{
				count++;
			}
		}

		fclose(fp);

		if (count < 1)
			count = 1;

		return count;
#elif __IOS__
		int count = 0;
		size_t len = sizeof(count);
		sysctlbyname("hw.ncpu", &count, &len, NULL, 0);

		if (count < 1)
			count = 1;

		return count;
#else
#ifdef _OPENMP
		return omp_get_max_threads();
#else
		return 1;
#endif // _OPENMP
#endif
	}

	int g_cpucount = get_cpucount();

	int get_cpu_count()
	{
		return g_cpucount;
	}

#ifdef __ANDROID__

	int get_max_freq_khz(int cpuid)
	{
		// first try, for all possible cpu
		char path[256];
		sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

		FILE* fp = fopen(path, "rb");

		if (!fp)
		{
			// second try, for online cpu
			sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuid);
			fp = fopen(path, "rb");

			if (fp)
			{
				int max_freq_khz = 0;
				while (!feof(fp))
				{
					int freq_khz = 0;
					int nscan = fscanf(fp, "%d %*d", &freq_khz);
					if (nscan != 1)
						break;

					if (freq_khz > max_freq_khz)
						max_freq_khz = freq_khz;
				}

				fclose(fp);

				if (max_freq_khz != 0)
					return max_freq_khz;

				fp = NULL;
			}

			if (!fp)
			{
				// third try, for online cpu
				sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
				fp = fopen(path, "rb");

				if (!fp)
					return -1;

				int max_freq_khz = -1;
				fscanf(fp, "%d", &max_freq_khz);

				fclose(fp);

				return max_freq_khz;
			}
		}

		int max_freq_khz = 0;
		while (!feof(fp))
		{
			int freq_khz = 0;
			int nscan = fscanf(fp, "%d %*d", &freq_khz);
			if (nscan != 1)
				break;

			if (freq_khz > max_freq_khz)
				max_freq_khz = freq_khz;
		}

		fclose(fp);

		return max_freq_khz;
	}

	int set_sched_affinity(const std::vector<int>& cpuids)
	{
		// cpu_set_t definition
		// ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))
		typedef struct
		{
			unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
		} cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) \
  memset((cpusetp), 0, sizeof(cpu_set_t))

		// set affinity for thread
#ifdef __GLIBC__
		pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
		pid_t pid = getpid();
#else
		pid_t pid = gettid();
#endif
#endif
		cpu_set_t mask;
		CPU_ZERO(&mask);
		for (int i = 0; i < (int)cpuids.size(); i++)
		{
			CPU_SET(cpuids[i], &mask);
		}

		int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
		if (syscallret)
		{
			fprintf(stderr, "syscall error %d\n", syscallret);
			return -1;
		}

		return 0;
	}

	int sort_cpuid_by_max_frequency(std::vector<int>& cpuids, int* little_cluster_offset)
	{
		const int cpu_count = cpuids.size();

		*little_cluster_offset = 0;

		if (cpu_count == 0)
			return 0;

		std::vector<int> cpu_max_freq_khz;
		cpu_max_freq_khz.resize(cpu_count);

		for (int i = 0; i < cpu_count; i++)
		{
			int max_freq_khz = get_max_freq_khz(i);

			// printf("%d max freq = %d khz\n", i, max_freq_khz);

			cpuids[i] = i;
			cpu_max_freq_khz[i] = max_freq_khz;
		}

		// sort cpuid as big core first
		// simple bubble sort
		for (int i = 0; i < cpu_count; i++)
		{
			for (int j = i + 1; j < cpu_count; j++)
			{
				if (cpu_max_freq_khz[i] < cpu_max_freq_khz[j])
				{
					// swap
					int tmp = cpuids[i];
					cpuids[i] = cpuids[j];
					cpuids[j] = tmp;

					tmp = cpu_max_freq_khz[i];
					cpu_max_freq_khz[i] = cpu_max_freq_khz[j];
					cpu_max_freq_khz[j] = tmp;
				}
			}
		}

		// SMP
		int mid_max_freq_khz = (cpu_max_freq_khz.front() + cpu_max_freq_khz.back()) / 2;
		if (mid_max_freq_khz == cpu_max_freq_khz.back())
			return 0;

		for (int i = 0; i < cpu_count; i++)
		{
			if (cpu_max_freq_khz[i] < mid_max_freq_khz)
			{
				*little_cluster_offset = i;
				break;
			}
		}

		return 0;
	}
#endif // __ANDROID__

	int g_powersave = 0;

	int get_cpu_powersave()
	{
		return g_powersave;
	}

	int set_cpu_powersave(int powersave)
	{
#ifdef __ANDROID__
		std::vector<int> sorted_cpuids;
		int little_cluster_offset = 0;

		if (sorted_cpuids.empty())
		{
			// 0 ~ g_cpucount
			sorted_cpuids.resize(g_cpucount);
			for (int i = 0; i < g_cpucount; i++)
			{
				sorted_cpuids[i] = i;
			}

			// descent sort by max frequency
			sort_cpuid_by_max_frequency(sorted_cpuids, &little_cluster_offset);
		}

		if (little_cluster_offset == 0 && powersave != 0)
		{
			powersave = 0;
			fprintf(stderr, "SMP cpu powersave not supported\n");
		}

		// prepare affinity cpuid
		std::vector<int> cpuids;
		if (powersave == 0)
		{
			cpuids = sorted_cpuids;
		}
		else if (powersave == 1)
		{
			cpuids = std::vector<int>(sorted_cpuids.begin() + little_cluster_offset, sorted_cpuids.end());
		}
		else if (powersave == 2)
		{
			cpuids = std::vector<int>(sorted_cpuids.begin(), sorted_cpuids.begin() + little_cluster_offset);
		}
		else
		{
			fprintf(stderr, "powersave %d not supported\n", powersave);
			return -1;
		}

#ifdef _OPENMP
		// set affinity for each thread
		int num_threads = cpuids.size();
		omp_set_num_threads(num_threads);
		std::vector<int> ssarets(num_threads, 0);
#pragma omp parallel for
		for (int i = 0; i < num_threads; i++)
		{
			ssarets[i] = set_sched_affinity(cpuids);
		}
		for (int i = 0; i < num_threads; i++)
		{
			if (ssarets[i] != 0)
			{
				return -1;
			}
		}
#else
		int ssaret = set_sched_affinity(cpuids);
		if (ssaret != 0)
		{
			return -1;
		}
#endif

		g_powersave = powersave;

		return 0;
#elif __IOS__
		// thread affinity not supported on ios
		return -1;
#else
		// TODO
		(void)powersave;  // Avoid unused parameter warning.
		return -1;
#endif
	}

#endif //!arm


	int get_omp_num_threads()
	{
#ifdef _OPENMP
		return omp_get_num_threads();
#else
		return 1;
#endif
	}

	void set_omp_num_threads(int num_threads)
	{
#ifdef _OPENMP
		omp_set_num_threads(num_threads);
#else
		(void)num_threads;
#endif
	}

	int get_omp_dynamic()
	{
#ifdef _OPENMP
		return omp_get_dynamic();
#else
		return 0;
#endif
	}

	void set_omp_dynamic(int dynamic)
	{
#ifdef _OPENMP
		omp_set_dynamic(dynamic);
#else
		(void)dynamic;
#endif
	}
}