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

LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

SRC_PATH := $(LOCAL_PATH)/../../../src/Longinus
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../../include/Longinus
LOCAL_C_INCLUDES += $(COMMON_INCLUDES)
LOCAL_MODULE := Longinus-java
LOCAL_SRC_FILES := $(SRC_PATH)/common.cpp $(SRC_PATH)/ImageOperation.cpp \
					$(SRC_PATH)/InternalLonginusCascade.cpp $(SRC_PATH)/LonginusDetector.cpp \
					$(SRC_PATH)/matcher.cpp Longinus-jni.cpp

ifeq ($(RELEASE_SDK), 0)
	LOCAL_SRC_FILES += tinyxml2.cpp
endif

LOCAL_CPPFLAGS += $(EXTRA_CPPFLAGS)

LOCAL_STATIC_LIBRARIES := Romancia

ifeq ($(TRIAL), 0)
	LOCAL_STATIC_LIBRARIES += Damocles
endif

LOCAL_LDLIBS += -L$(OPENCV_ANDROID_SDK_ROOT)/sdk/native/libs/$(APP_ABI) -lopencv_java3

include $(BUILD_SHARED_LIBRARY)
