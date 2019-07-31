# tensor_builder::to_tensor_float_shared Member Function

## Information

<table>
    <tr>
    <td>

Header
    </td>
    <td>

tensor_builder.hpp
    </td>
    </tr>
    <tr>
    <td>

Namespace
    </td>
    <td>

**```glasssix::excalibur```**
    </td>
    </tr>
    <tr>
    <td>

Description
    </td>
    <td>

Allocates a block on the heap and converts the loaded bitmap to a floating-point tensor.
    </td>
    </tr>
</table>

<br>

## to_tensor_float_shared(tensor_layout)

Allocates a **```std::shared_ptr<tensor<float>>```**  which it converts the loaded bitmap to.

```C++
virtual std::shared<tensor<float>> to_tensor_float_shared(tensor_layout layout) = 0;
```

### Parameters

<dl>
    <dt>

**layout**
    </dt>
    <dd>

[**```glasssix::excalibur::tensor_layout```**](../tensor_layout.md)
    </dd>
</dl>

Indicates the channel count and the pixel format.

### Returns

**```std::shared_ptr<glasssix::tensor<float>>```**

A smart pointer known as **```std::shared_ptr<glasssix::tensor<float>>```**, which contains the converted tensor if the operation was successful and otherwise, an null pointer is included.

<br>

## Links
<a href="../../Tensor_IO_Manual_en_US.md"><img src="../../images/home.png" width="32" height="32"></img></a>