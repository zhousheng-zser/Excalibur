# Copyright (C) 2009 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
RELEASE_SDK := 1
TRIAL := 0
QUANTIZATION_TYPE := SINGLE
COSINE_DISTANCE := 1
USE_OPENMP := 0
USE_OPENCV := 0

COMMON_INCLUDES := C:/Tools/vcpkg/installed/x64-windows/include
OPENCV_ANDROID_SDK_ROOT := E:/OpenCV-android-sdk
COMMON_INCLUDES += $(OPENCV_ANDROID_SDK_ROOT)/sdk/native/jni/include

ifeq ($(USE_OPENMP), 1)
ifeq ($(NDK_TOOLCHAIN_VERSION), clang)
	EXTRA_CPPFLAGS += -openmp
else
	EXTRA_CPPFLAGS += -fopenmp
endif	
endif

ifeq ($(TRIAL), 1)
	EXTRA_CPPFLAGS += -DTRIAL
endif

ifeq ($(RELEASE_SDK), 1)
	EXTRA_CPPFLAGS += -DRELEASE_SDK
endif

ifeq ($(USE_OPENCV), 1)
	EXTRA_CPPFLAGS += -DUSE_OPENCV
endif

EXTRA_CPPFLAGS += -DFMT_HEADER_ONLY

include $(call all-subdir-makefiles)

