# operator| 运算符

## 信息

<table>
    <tr>
    <td>

头文件
    </td>
    <td>

tensor_conversions.hpp
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

支持 **```glasssix::tensor<T>```** 的基础类型、像素及通道布局转换功能。
    </td>
    </tr>
</table>

## 重载

<table>
    <tr>
    <td>

**[template<typename TSource, typename TDestination><br>operator|(const tensor\<TSource\>&, const tensor_convert_to_tag\<TDestination\>&)](#templatetypename-tsource-typename-tdestinationbroperatorconst-tensortsource-const-tensorconverttotagtdestination)**
    </td>
    <td>

转换 **```glasssix::tensor<T>```** 的基础类型。
    </td>
    </tr>
    <tr>
    <td>

**[template<typename TSource, typename TDestination><br>operator|(const std::shared_ptr\<tensor\<TSource\>\>&, const tensor_convert_to_tag\<TDestination\>&)](#templatetypename-tsource-typename-tdestinationbroperatorconst-stdsharedptrtensortsource-const-tensorconverttotagtdestination)**
    </td>
    <td>

转换 **```std::shared_ptr<glasssix::tensor<T>>```** 的基础类型。
    </td>
    </tr>
    <tr>
    <td>

**[template<typename TSource, tensor_layout layout><br>operator|(const tensor\<TSource\>&, const tensor_convert_layout_to_tag\<layout\>&)](#templatetypename-tsource-tensorlayout-layoutbroperatorconst-tensortsource-const-tensorconvertlayouttotaglayout)**
    </td>
    <td>

转换 **```glasssix::tensor<T>```** 的像素及通道格式。
    </td>
    </tr>
    <tr>
    <td>

**[template<typename TSource, tensor_layout layout><br>operator|(const std::shared_ptr\<tensor\<TSource\>\>&, const tensor_convert_layout_to_tag\<layout\>&)](#templatetypename-tsource-tensorlayout-layoutbroperatorconst-stdsharedptrtensortsource-const-tensorconvertlayouttotaglayout)**
    </td>
    <td>

转换 **```std::shared_ptr<glasssix::tensor<T>>```** 的像素及通道格式。
    </td>
    </tr>
</table>

<br>

## template<typename TSource, typename TDestination><br>operator|(const tensor\<TSource\>&, const tensor_convert_to_tag\<TDestination\>&)

转换 **```glasssix::tensor<T>```** 的基础类型。

```C++
template<typename TSource, typename TDestination>
inline tensor<TDestination> operator|(const tensor<TSource>& source, const tensor_convert_to_tag<TDestination>& tag);
```

### 模板参数

**```TSource```**

源基础类型。

**```TDestination```**

目标基础类型。

### 参数

<dl>
    <dt>

**source**
    </dt>
    <dd>

**```const glasssix::tensor<TSource>&```**
    </dd>
</dl>

源张量。

<dl>
    <dt>

**tag**
    </dt>
    <dd>

[**```const tensor_convert_to_tag<TDestination>&```**](tensor_convert_to_tag.md)
    </dd>
</dl>

包含目标基础类型信息的标记。

### 返回

**```glasssix::tensor<TDestination>```**

目标张量。

<br>

## template<typename TSource, typename TDestination><br>operator|(const std::shared_ptr\<tensor\<TSource\>\>&, const tensor_convert_to_tag\<TDestination\>&

转换 **```std::shared_ptr<glasssix::tensor<T>>```** 的基础类型。

```C++
template<typename TSource, typename TDestination>
inline std::shared_ptr<tensor<TDestination>> operator|(const std::shared_ptr<tensor<TSource>>& source, const tensor_convert_to_tag<TDestination>& tag);
```

### 模板参数

**```TSource```**

源基础类型。

**```TDestination```**

目标基础类型。

### 参数

<dl>
    <dt>

**source**
    </dt>
    <dd>

**```std::shared_ptr<glasssix::tensor<TSource>>&```**
    </dd>
</dl>

源张量。

<dl>
    <dt>

**tag**
    </dt>
    <dd>

[**```const tensor_convert_to_tag<TDestination>&```**](tensor_convert_to_tag.md)
    </dd>
</dl>

包含目标基础类型信息的标记。

### 返回

**```std::shared_ptr<glasssix::tensor<TDestination>>```**

目标张量。

<br>

## template<typename TSource, tensor_layout layout><br>operator|(const tensor\<TSource\>&, const tensor_convert_layout_to_tag\<layout\>&)

转换 **```glasssix::tensor<T>```** 的像素及通道布局。

```C++
template<typename TSource, tensor_layout layout>
inline tensor<TSource> operator|(const tensor<TSource>& source, const tensor_convert_layout_to_tag<layout>& tag);
```

### 模板参数

**```TSource```**

源基础类型。

<dl>
    <dt>

**layout**
    </dt>
    <dd>

**```tensor_layout```**
    </dd>
</dl>

目标像素及通道布局。

### 参数

<dl>
    <dt>

**source**
    </dt>
    <dd>

**```const glasssix::tensor<TSource>&```**
    </dd>
</dl>

源张量。

<dl>
    <dt>

**tag**
    </dt>
    <dd>

[**```const tensor_convert_layout_to_tag<layout>&```**](tensor_convert_layout_to_tag.md)
    </dd>
</dl>

包含目标像素及通道类型信息的标记。

### 返回

**```glasssix::tensor<TDestination>```**

目标张量。

<br>


## template<typename TSource, tensor_layout layout><br>operator|(const std::shared_ptr\<tensor\<TSource\>\>&, const tensor_convert_layout_to_tag\<layout\>&)

转换 **```glasssix::tensor<T>```** 的像素及通道布局。

```C++
template<typename TSource, tensor_layout layout>
inline std::shared_ptr<tensor<TSource>> operator|(const std::shared_ptr<tensor<TSource>>& source, const tensor_convert_layout_to_tag<layout>& tag)
```

### 模板参数

**```TSource```**

源基础类型。

<dl>
    <dt>

**layout**
    </dt>
    <dd>

**```tensor_layout```**
    </dd>
</dl>

目标像素及通道布局。

### 参数

<dl>
    <dt>

**source**
    </dt>
    <dd>

**```std::shared_ptr<glasssix::tensor<TSource>>&```**
    </dd>
</dl>

源张量。

<dl>
    <dt>

**tag**
    </dt>
    <dd>

[**```const tensor_convert_layout_to_tag<layout>&```**](tensor_convert_layout_to_tag.md)
    </dd>
</dl>

包含目标像素及通道类型信息的标记。

### 返回

**```std::shared_ptr<glasssix::tensor<TDestination>>```**

目标张量。

<br>

## 链接
<a href="../../Tensor_IO_Manual_zh_CN.md"><img src="../../images/home.png" width="32" height="32"></img></a>
