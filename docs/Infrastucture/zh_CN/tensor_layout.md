# tensor_layout 枚举

## 信息

<table>
    <tr>
    <td>

头文件
    </td>
    <td>

tensor_layout.hpp
    </td>
    </tr>
    <tr>
    <td>

命名空间
    </td>
    <td>

**```glasssix::excalibur```**
    </td>
    </tr>
    <tr>
    <td>

功能描述
    </td>
    <td>

定义标准张量 **```glasssix::tensor<T>```** 的内存布局。
    </td>
    </tr>
</table>

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

<table>
    <tr>
    <td>

rgb
    </td>
    <td>

三通道真彩色 RGB 图像。
    </td>
    </tr>
    <tr>
    <td>

rgba
    </td>
    <td>

四通道真彩色 RGBA 图像，其中包括一个 Alpha 通道。
    </td>
    </tr>
    <tr>
    <td>

grayscale
    </td>
    <td>

单通道灰度图。
    </td>
    </tr>
    <tr>
    <td>

grayscale_3
    </td>
    <td>

三通道灰度图。对单个像素而言，每个通道的取值相同。
    </td>
    </tr>
</table>

## 链接
<a href="../Tensor_IO_Manual_zh_CN.md"><img src="../images/home.png" width="32" height="32"></img></a>