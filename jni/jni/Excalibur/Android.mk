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

LOCAL_PATH := $(call my-dir)/../../../src/Excalibur
include $(CLEAR_VARS)

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../include/Excalibur $(LOCAL_PATH)/../../include
LOCAL_C_INCLUDES += $(COMMON_INCLUDES)
LOCAL_MODULE := Excalibur
LOCAL_SRC_FILES := axpy.cpp bn.cpp concat.cpp conv_native_cpu.cpp \
					conv_1x1s1_cpu.cpp \
					conv_winograd_cpu.cpp convdw_3x3s1_cpu.cpp convdw_3x3s2_cpu.cpp \
					deconv.cpp eltwise.cpp \
					flip.cpp hswish.cpp im2col.cpp inner_product.cpp \
					io.cpp math_functions.cpp mirrormax.cpp \
					normalize.cpp pca.cpp pooling.cpp \
					prelu.cpp reshape.cpp scale.cpp sigmoid.cpp slice.cpp \
					softmax.cpp tensor_operation_cpu.cpp upsample.cpp \
					arm/batchnorm_arm.cpp arm/conv_arm.cpp \
					arm/deconv_arm.cpp arm/eltwise_arm.cpp \
					arm/inner_product_arm.cpp arm/pooling_arm.cpp \
					arm/prelu_arm.cpp arm/scale_arm.cpp \
					arm/sigmoid_arm.cpp arm/softmax_arm.cpp
					

LOCAL_CPPFLAGS += $(EXTRA_CPPFLAGS)
LOCAL_STATIC_LIBRARIES := Julius Primitives
include $(BUILD_STATIC_LIBRARY)
