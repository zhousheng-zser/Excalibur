# Julius

A SIMD supported [BLAS](http://www.netlib.org/blas/) library. Now only some important interfaces were implementated. Given our usage, there is no plans for complex value supporting(i.e. only *float* and *double* datatypes will be implemented) currently.

## Implementation Status

| Function |Native Code|  SSE	 |	AVX   |	AVX512|
|:--------:|:---------:|:-------:|:------:|:-----:|
| *sdsdot* |   TP  |  F |    F  |  F |
| *dsdot* |   F  |  F |    F  |  F |
| *s/ddot* |   TP/F  |  F |    F  |  F |
| *s/daxpby* |   TP/F  |  F |    F  |  F |
| *s/dgemv* |   TP/F  |  TP/F |   TP/F  |  F |
| *s/dgemm* |   TP/F  |  TP/F |   TP/F  |  F |


- TP: Implementated and test passed;
- F: Not implementated;
- TE: Implementated but error exists;
- --: No implementation;
- NT: Implementated but no test yet;
  
## Performance Status

- OS: Windows10 v1803(x64)
- CPU: Intel Core i7-8700k
- Memory: 32GB DDR4 2400MHz
- Compiler: MSVC v1910

### ?gemv


### ?gemm
