# tensor_builder::to_tensor_uint8 成员函数

## 信息

<table>
    <tr>
    <td>

头文件
    </td>
    <td>

tensor_builder.hpp
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

将已加载的标准位图转换为字节型张量。
    </td>
    </tr>
</table>

<br>

## to_tensor_uint8(tensor_layout)

将已加载的标准位图转换为字节型张量。

```C++
virtual std::optional<tensor<uint8_t>> to_tensor_uint8(tensor_layout layout) = 0;
```

### 参数

<dl>
    <dt>

**layout**
    </dt>
    <dd>

[**```glasssix::excalibur::tensor_layout```**](../tensor_layout.md)
    </dd>
</dl>

描述通道数及像素格式的值。

### 返回

**```std::optional<glasssix::tensor<uint8_t>>```**

一个可选值。若操作成功，则为生成的张量；否则为空。

<br>

## 链接
<a href="../../Tensor_IO_Manual_zh_CN.md"><img src="../../images/home.png" width="32" height="32"></img></a>