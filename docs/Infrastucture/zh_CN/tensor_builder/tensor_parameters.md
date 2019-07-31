# tensor_builder::tensor_parameters 成员函数

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

设置标准张量的参数。
    </td>
    </tr>
</table>

## 重载

<table>
    <tr>
    <td>

**[tensor_parameters(orderType)](#tensor_parametersorderType)**
    </td>
    <td>

设置标准张量的参数。
    </td>
    </tr>
    <tr>
    <td>

**[tensor_parameters(orderType, int)](#tensor_parametersorderType-int)**
    </td>
    <td>

设置标准张量的参数。
    </td>
    </tr>
</table>

<br>

## tensor_parameters(orderType)

设置标准张量的参数。

```C++
virtual void tensor_parameters(orderType order) = 0;
```

### 参数

<dl>
    <dt>

**order**
    </dt>
    <dd>

**```glasssix::orderType```**
    </dd>
</dl>

内存布局，满足 **```glasssix::tensor```** 标准格式。

<br>

## tensor_parameters(orderType, int)

设置标准张量的参数。

```C++
virtual void tensor_parameters(orderType order, int device) = 0;
```
### 参数

<dl>
    <dt>

**order**
    </dt>
    <dd>

**```glasssix::orderType```**
    </dd>
</dl>

内存布局，满足 **```glasssix::tensor```** 标准格式。

<dl>
    <dt>

**device**
    </dt>
    <dd>

**```int```**
    </dd>
</dl>

表示一个物理设备，用于存放张量的数据。

|取值|描述|
|---|---|
|-1|使用 CPU 存储。|
|>= 0|代表一个 GPU 的编号。|

<br>

## 链接
<a href="../../Tensor_IO_Manual_en_US.md"><img src="../../images/home.png" width="32" height="32"></img></a>