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

SRC_PATH := $(LOCAL_PATH)/../../../src/Irisviel
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../../include/Irisviel $(LOCAL_PATH)/../../../include
LOCAL_C_INCLUDES += $(COMMON_INCLUDES)
LOCAL_MODULE := Irisviel
LOCAL_SRC_FILES := $(SRC_PATH)/database_business_wrapper.cpp $(SRC_PATH)/database_cache.cpp $(SRC_PATH)/database_feature_observer.cpp \
					$(SRC_PATH)/database_header.cpp $(SRC_PATH)/database_manager.cpp $(SRC_PATH)/database_record.cpp \
					$(SRC_PATH)/distance.cpp $(SRC_PATH)/face_service.cpp $(SRC_PATH)/filesystem_utils.cpp $(SRC_PATH)/index_builder.cpp \
					$(SRC_PATH)/irisviel_c.cxx $(SRC_PATH)/irisviel_search.cpp $(SRC_PATH)/kgraph_internal.cpp \
					$(SRC_PATH)/memory_mapping.cpp $(SRC_PATH)/memory_mapping_operator.cpp $(SRC_PATH)/ngraph_internal.cpp \
					$(SRC_PATH)/search.cpp 

LOCAL_CPPFLAGS += $(EXTRA_CPPFLAGS)
ifeq ($(COSINE_DISTANCE), 1)
	LOCAL_CPPFLAGS += -DCOSINE_DISTANCE
endif

LOCAL_STATIC_LIBRARIES := Primitives

include $(BUILD_SHARED_LIBRARY)
