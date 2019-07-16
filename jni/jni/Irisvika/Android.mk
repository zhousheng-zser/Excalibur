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

SRC_PATH := $(LOCAL_PATH)/../../../src/Irisvian
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../../include/Irisvian
LOCAL_C_INCLUDES += $(COMMON_INCLUDES)
LOCAL_MODULE := Irisvian-java
LOCAL_SRC_FILES := $(SRC_PATH)/distance.cpp $(SRC_PATH)/index.cpp $(SRC_PATH)/IrisvianSearch.cpp \
					$(SRC_PATH)/kGraph.cpp $(SRC_PATH)/nGraph.cpp $(SRC_PATH)/search.cpp \
					Irisvian-jni.cpp

LOCAL_CPPFLAGS += $(EXTRA_CPPFLAGS)

include $(BUILD_SHARED_LIBRARY)
