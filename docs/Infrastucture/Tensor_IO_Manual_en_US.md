# Tensor I/O Manual

## Revisions
|Content|Date|Author|
|---|---|---|
|Origin|2019.07.29 Mon.|Zhang Mingyu|

## Introduction
Tensor I/O is a library for input/output operations on a standard tensor known as **```glasssix::tensor<T>```** within the **Glassix Common Headers**. It is easy-to-use for library developers and end users to handle conversions between standard bitmap files (i.e. .jpg, .bmp, .png, .tiff) and standard tensors. Besides, some common pixel converters are packed alongside other functions.

## Consuming
There are 3 lite header files for users to consume this library conveniently.

|Header|Function|
|---|---|
|tensor_builder.hpp|Include this for interops between standard bitmap files and standard tensors.|
|tensor_builder_factory.hpp|Include this for interops between standard bitmap files and standard tensors.|
|tensor_conversions.hpp|Include this for pixel conversions on standard tensors.|

The header files listed above can be located at /include/Infrastructure folder in your authorized SDK. The including directory of your build tooling should be set accordingly.

## Interface List
- Namespace **```glasssix::excalibur```**
   
   - **Common**
      - [**```tensor_layout```**](en_US/tensor_layout.md) Enum

   - **Standard Bitmap I/O** Module
      - [**```tensor_builder```**]() Interface
      - [**```tensor_builder_factory```**]() Class Factory
      - [**```tensor_builder_implementation```**]() Enum

   - **Standard Tensor Pixel Conversions** Module
      - [**```tensor_convert_to_tag```**]() Class
      - [**```tensor_convert_layout_to_tag```**]() Class
      - [**```tensor_convert_to```**]() Global Variable
      - [**```tensor_convert_layout_to```**]() Global Variable
      - [**```operator|```**]() Operator

## Contributors
- Zhang Mingyu / Glasssix