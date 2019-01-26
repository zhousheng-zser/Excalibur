# Romancia
A face detection Library.

## How to compile
### windows(vs2017)
Open Romancia.vcxproj, some explicit options would be manual setting.

### linux
#### x86
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_OPENMP=ON -DTENSOR_INCLUDE_DIR=<your tensor include directory> -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<director to be install>
#### arm
mkdir build
cd build
cmake .. -Dplatform=ARM -DUSE_OPENMP=ON -DTENSOR_INCLUDE_DIR=<your tensor include directory> -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<director to be install>

## Internal or External SDK
Default outpu external SDK. If you want internal SDK, adding defition '-DInternal_SDK=ON'

## Hardcode transform
If you want transform xml model to hardcode mode, adding definition '-DInternal_SDK=ON -DHARDCODE_TRANSFORM=ON'