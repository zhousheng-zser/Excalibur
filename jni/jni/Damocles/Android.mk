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
	
	SRC_PATH := $(LOCAL_PATH)/../../../src/Damocles

	LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../../include/Damocles $(LOCAL_PATH)/../../../include
	LOCAL_C_INCLUDES += $(COMMON_INCLUDES)
	LOCAL_MODULE := Damocles
	LOCAL_SRC_FILES := $(SRC_PATH)/damocles.cpp $(SRC_PATH)/mtcnn.cpp \
						$(SRC_PATH)/mtcnn_onet.cpp $(SRC_PATH)/mtcnn_pnet.cpp $(SRC_PATH)/mtcnn_rnet.cpp \
						$(SRC_PATH)/mtcnn_mobile.cpp $(SRC_PATH)/mtcnn_mobile_nir.cpp $(SRC_PATH)/onet_mobile_nir.cpp \
						$(SRC_PATH)/pnet_mobile_nir.cpp $(SRC_PATH)/rnet_mobile_nir.cpp \
						$(SRC_PATH)/onet_mobile.cpp $(SRC_PATH)/pnet_mobile.cpp $(SRC_PATH)/rnet_mobile.cpp \

	LOCAL_CPPFLAGS += $(EXTRA_CPPFLAGS)
	LOCAL_STATIC_LIBRARIES := Excalibur
	include $(BUILD_STATIC_LIBRARY)
endif
