# tensor_builder::from_tensor Member Function

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

Converts a tensor to a standard bitmap.
    </td>
    </tr>
</table>

## Overloads

<table>
    <tr>
    <td>

**[from_tensor(const tensor<float>&, tensor_layout)](#from_tensorconst-tensorfloat-tensor_layout)**
    </td>
    <td>

Converts a floating-point tensor to a standard bitmap.
    </td>
    </tr>
    <tr>
    <td>

**[from_tensor(const tensor<uint8_t>&, tensor_layout)](#from_tensorconst-tensor<uint8_t-tensor_layout)**
    </td>
    <td>

Converts a uint8 tensor to a standard bitmap.
    </td>
    </tr>
</table>

<br>

## from_tensor(const tensor<float>&, tensor_layout)

Converts a floating-point tensor to a standard bitmap.

```C++
virtual bool from_tensor(const tensor<float>& data, tensor_layout layout) = 0;
```

### Parameters

<dl>
    <dt>

**data**
    </dt>
    <dd>

**```const glasssix::tensor<float>&```**
    </dd>
</dl>

The floating-point tensor.

<dl>
    <dt>

**layout**
    </dt>
    <dd>

[**```glasssix::excalibur::tensor_layout```**](../tensor_layout.md)
    </dd>
</dl>

Indicates the channel count and the pixel format.

## Returns

**```bool```**

**```true```** if the operation was successful; otherwise, **```false```** .

<br>

## from_tensor(const tensor<uint8_t>&, tensor_layout)

Converts a uint8 tensor to a standard bitmap.

```C++
virtual bool from_tensor(const tensor<uint8_t>& data, tensor_layout layout) = 0;
```
### Parameters

<dl>
    <dt>

**data**
    </dt>
    <dd>

**```const glasssix::tensor<uint8_t>&```**
    </dd>
</dl>

The uint8 tensor.

<dl>
    <dt>

**layout**
    </dt>
    <dd>

[**```glasssix::excalibur::tensor_layout```**](../tensor_layout.md)
    </dd>
</dl>

Indicates the channel count and the pixel format.

## Returns

**```bool```**

**```true```** if the operation was successful; otherwise, **```false```** .

<br>

## Links
<a href="../../Tensor_IO_Manual_en_US.md"><img src="../../images/home.png" width="32" height="32"></img></a>