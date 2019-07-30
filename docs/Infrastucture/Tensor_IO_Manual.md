# Tensor I/O 使用手册

## 修订记录
|内容|时间|作者|
|---|---|---|
|编撰第一版|2019.07.29 Mon.|张铭雨|

## 简介
Tensor I/O 是一款基于开源图像处理类库开发的标准张量（**```glasssix::tensor<T>```**）输入输出模块。它为库开发者和最终用户提供了易用的接口，以处理标准图像文件（.jpg、.bmp、.png、.tiff）和标准张量间的数据转换，并提供了常用的像素转换函数。

## 使用
Tensor I/O 为用户提供了 3 个轻量级接口头文件，以方便地使用本库的各项功能。

|文件|功能|
|---|---|
|tensor_builder.hpp|使用标准图像文件和标准张量互操作功能时包含此文件。|
|tensor_builder_factory.hpp|使用标准图像文件和标准张量互操作功能时包含此文件。|
|tensor_convertions.hpp|使用标准张量像素转换功能时包含此文件。|

以上头文件包含在您获得的开发包的 /include/Infrastructure 目录，请结合您使用的构建工具设置相应的包含目录。

## 接口列表
- 命名空间 **```glasssix::excalibur```**
   
   - **公共模块**
      - [**```tensor_layout```**]() 枚举

   - **标准图像输入输出** 功能模块
      - [**```tensor_builder```**]() 接口类
      - [**```tensor_builder_factory```**]() 工厂类
      - [**```tensor_builder_implementation```**]() 枚举

   - **标准张量像素转换** 功能模块
      - [**```tensor_convert_to_tag```**]() 类
      - [**```tensor_convert_layout_to_tag```**]() 类
      - [**```tensor_convert_to```**]() 全局变量
      - [**```tensor_convert_layout_to```**]() 全局变量
      - [**```operator|```**]() 运算符

## 贡献者
- Zhang Mingyu / Glassix