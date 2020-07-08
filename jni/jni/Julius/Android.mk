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

LOCAL_PATH := $(call my-dir)/../../../src/Julius
include $(CLEAR_VARS)

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../include/Julius $(LOCAL_PATH)/../../include
LOCAL_C_INCLUDES += $(COMMON_INCLUDES)
LOCAL_MODULE := Julius
LOCAL_SRC_FILES := julius.cpp julius_asum.cpp julius_axpby.cpp \
					julius_dot.cpp julius_gemm.cpp julius_gemm_align.cpp julius_gemm_auto.cpp julius_gemv.cpp \
					julius_nrm2.cpp julius_scal.cpp julius_sdot.cpp

LOCAL_CPPFLAGS += $(EXTRA_CPPFLAGS)
include $(BUILD_STATIC_LIBRARY)
