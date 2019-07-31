# tensor_builder::from_tensor 成员函数

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

将一个张量转换为标准位图。
    </td>
    </tr>
</table>

## 重载

<table>
    <tr>
    <td>

**[from_tensor(const tensor<float>&, tensor_layout)](#from_tensorconst-tensorfloat-tensor_layout)**
    </td>
    <td>

将一个浮点型张量转换为标准位图。
    </td>
    </tr>
    <tr>
    <td>

**[from_tensor(const tensor<uint8_t>&, tensor_layout)](#from_tensorconst-tensoruint8_t-tensor_layout)**
    </td>
    <td>

将一个字节型张量转换为标准位图。
    </td>
    </tr>
</table>

<br>

## from_tensor(const tensor<float>&, tensor_layout)

将一个浮点型张量转换为标准位图。

```C++
virtual bool from_tensor(const tensor<float>& data, tensor_layout layout) = 0;
```

### 参数

<dl>
    <dt>

**data**
    </dt>
    <dd>

**```const glasssix::tensor<float>&```**
    </dd>
</dl>

浮点型张量。

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

**```bool```**

如果操作成功，则为 **```true```**；否则为 **```false```**。

<br>

## from_tensor(const tensor<uint8_t>&, tensor_layout)

将一个字节型张量转换为标准位图。

```C++
virtual bool from_tensor(const tensor<uint8_t>& data, tensor_layout layout) = 0;
```
### 参数

<dl>
    <dt>

**data**
    </dt>
    <dd>

**```const glasssix::tensor<uint8_t>&```**
    </dd>
</dl>

字节型张量。

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

**```bool```**

如果操作成功，则为 **```true```**；否则为 **```false```**。

<br>

## 链接
<a href="../../Tensor_IO_Manual_zh_CN.md"><img src="../../images/home.png" width="32" height="32"></img></a>