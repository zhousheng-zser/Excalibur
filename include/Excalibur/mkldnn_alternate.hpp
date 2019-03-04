#pragma once
#ifndef _MKLDNN_ALTERNATE_HPP_
#define _MKLDNN_ALTERNATE_HPP_
#include "mkl_alternate.hpp"

#ifdef USE_MKL 
#define STR1(x) #x
#define STR(x) STR1(x)

#define CHECK_MKL(dnnCall) do { \
    dnnError_t e = dnnCall; \
    if (e != E_SUCCESS) { \
        printf("[%s:%d] %s = %d\n", __FILE__, __LINE__, STR(dnnCall), e); \
        throw std::runtime_error(STR(dnnCall)); \
    } \
} while (0)

#ifdef USE_MKLDNN



#endif //USE_MKLDNN
#endif //USE_MKL
#endif // _MKLDNN_ALTERNATE_HPP_