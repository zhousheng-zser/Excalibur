# tensor_builder::to_tensor_float_shared 成员函数

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

在堆上开辟空间，将已加载的标准位图转换为浮点型张量。
    </td>
    </tr>
</table>

<br>

## to_tensor_float_shared(tensor_layout)

申请一个 **```std::shared_ptr<tensor<float>>```**  对象并将已加载的标准位图转换为浮点型张量，存储于该对象中。

```C++
virtual std::shared<tensor<float>> to_tensor_float_shared(tensor_layout layout) = 0;
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

**```std::shared_ptr<glasssix::tensor<float>>```**

一个 **```std::shared_ptr<glasssix::tensor<float>>```** 智能指针。若操作成功，则为生成的张量；否则为空指针。

<br>

## 链接
<a href="../../Tensor_IO_Manual_zh_CN.md"><img src="../../images/home.png" width="32" height="32"></img></a>