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

ifeq ($(TRIAL), 0)
	LOCAL_PATH := $(call my-dir)
	include $(CLEAR_VARS)

	SRC_PATH := $(LOCAL_PATH)/../../../src/Primitives

	LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../../include/Primitives
	LOCAL_C_INCLUDES += $(COMMON_INCLUDES)
	LOCAL_MODULE := Primitives
	LOCAL_SRC_FILES := $(SRC_PATH)/cpu.cpp $(SRC_PATH)/gpu.cpp \
						$(SRC_PATH)/syncedmem.cpp $(SRC_PATH)/memory.cpp \
						$(SRC_PATH)/tensor.cpp $(SRC_PATH)/simd_types.cpp \
						$(SRC_PATH)/abi/component_loader.cpp \
						$(SRC_PATH)/abi/param_string.cpp \
						$(SRC_PATH)/abi/platform_encoding.cpp

	LOCAL_CPPFLAGS += $(EXTRA_CPPFLAGS)

	include $(BUILD_SHARED_LIBRARY)

endif