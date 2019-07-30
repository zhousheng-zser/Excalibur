# tensor_layout 枚举

## 信息

|   |   |
|---|---|
|头文件|tensor_layout.hpp|
|命名空间|**```glasssix::excalibur```**|
|功能描述|定义标准张量 **```glasssix::tensor<T>```** 的内存布局。

## 语法

```C++
enum class tensor_layout
{
    rgb,
    rgba,
    grayscale,
    grayscale_3
};
```

## 常量

|   |   |
|---|---|
|rgb|三通道真彩色 RGB 图像。|
|rgba|四通道真彩色 RGBA 图像，其中包括一个 Alpha 通道。|
|grayscale|单通道灰度图。|
|grayscale_3|三通道灰度图。对单个像素而言，每个通道的取值相同。|

## [返回](../Tensor_IO_Manual_zh_CN.md)