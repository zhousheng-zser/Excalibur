#ifndef SATURATE_HPP
#define SATURATE_HPP

/* Minimum and maximum values a `signed short int' can hold.  */
# define SHRT_MIN (-32768)
# define SHRT_MAX	32767
# define MAX(a,b)  ((a) < (b) ? (b) : (a))
# define USHRT_MAX (SHRT_MAX * 2 + 1)
# define UCHAR_MAX  0xff

namespace glasssix
{
    template<typename _Tp> static inline _Tp saturate_cast(int v)      { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(float v)    { return _Tp(v); }
    template<typename _Tp> static inline _Tp saturate_cast(uint8_t v)    { return _Tp(v); }

    inline int cvRound(float value) { return (int)(value + (value >= 0 ? 0.5f : -0.5f)); }

    template<> inline short saturate_cast<short>(int v)     { return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
    template<> inline short saturate_cast<short>(float v)        { int iv = cvRound(v); return saturate_cast<short>(iv); }
    template<> inline uint8_t saturate_cast<uint8_t>(int v)          { return (uint8_t)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
}
#endif