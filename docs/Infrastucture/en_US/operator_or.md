# operator| Operator

## Information

<table>
    <tr>
    <td>

Header
    </td>
    <td>

tensor_conversions.hpp
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

Supports conversions of the underlying type and the layout on a **```glasssix::tensor<T>```** .
    </td>
    </tr>
</table>

## Overloads

<table>
    <tr>
    <td>

**[template<typename TSource, typename TDestination><br>operator|(const tensor\<TSource\>&, const tensor_convert_to_tag\<TDestination\>&)](#templatetypename-tsource-typename-tdestinationoperatorconst-tensortsource-const-tensor_convert_to_tagtdestination)**
    </td>
    <td>

Converts the underlying type of a **```glasssix::tensor<T>```** .
    </td>
    </tr>
    <tr>
    <td>

**[template<typename TSource, typename TDestination><br>operator|(const std::shared_ptr\<tensor\<TSource\>\>&, const tensor_convert_to_tag\<TDestination\>&)](#templatetypename-tsource-typename-tdestinationoperatorconst-stdshared_ptrtensortsource-const-tensor_convert_to_tagtdestination)**
    </td>
    <td>

Converts the underlying type of a **```std::shared_ptr<glasssix::tensor<T>>```** .
    </td>
    </tr>
    <tr>
    <td>

**[template<typename TSource, tensor_layout layout><br>operator|(const tensor\<TSource\>&, const tensor_convert_layout_to_tag\<layout\>&)](#templatetypename-tsource-tensor_layout-layoutoperatorconst-tensortsource-const-tensor_convert_layout_to_taglayout)**
    </td>
    <td>

Converts the layout of a **```glasssix::tensor<T>```** .
    </td>
    </tr>
    <tr>
    <td>

**[template<typename TSource, tensor_layout layout><br>operator|(const std::shared_ptr\<tensor\<TSource\>\>&, const tensor_convert_layout_to_tag\<layout\>&)](#templatetypename-tsource-tensor_layout-layoutoperatorconst-stdshared_ptrtensortsource-const-tensor_convert_layout_to_taglayout)**
    </td>
    <td>

Converts the layout of a **```std::shared_ptr<glasssix::tensor<T>>```** .
    </td>
    </tr>
</table>

<br>

## template<typename TSource, typename TDestination><br>operator|(const tensor\<TSource\>&, const tensor_convert_to_tag\<TDestination\>&)

Converts the underlying type of a **```glasssix::tensor<T>```** .

```C++
template<typename TSource, typename TDestination>
inline tensor<TDestination> operator|(const tensor<TSource>& source, const tensor_convert_to_tag<TDestination>& tag);
```

### Template Parameters

**```TSource```**

The source underlying type.

**```TDestination```**

The destination underlying type.

### Parameters.

<dl>
    <dt>

**source**
    </dt>
    <dd>

**```const glasssix::tensor<TSource>&```**
    </dd>
</dl>

The source tensor.

<dl>
    <dt>

**tag**
    </dt>
    <dd>

[**```const tensor_convert_to_tag<TDestination>&```**](tensor_convert_to_tag.md)
    </dd>
</dl>

The tag containing the destination underlying type.

### Returns

**```glasssix::tensor<TDestination>```**

The destination tensor.

<br>

## template<typename TSource, typename TDestination><br>operator|(const std::shared_ptr\<tensor\<TSource\>\>&, const tensor_convert_to_tag\<TDestination\>&

Converts the underlying type of a **```std::shared_ptr<glasssix::tensor<T>>```** .

```C++
template<typename TSource, typename TDestination>
inline std::shared_ptr<tensor<TDestination>> operator|(const std::shared_ptr<tensor<TSource>>& source, const tensor_convert_to_tag<TDestination>& tag);
```

### Template Parameters.

**```TSource```**

The source underlying type.

**```TDestination```**

The destination underlying type.

### Parameters.

<dl>
    <dt>

**source**
    </dt>
    <dd>

**```std::shared_ptr<glasssix::tensor<TSource>>&```**
    </dd>
</dl>

The source tensor.

<dl>
    <dt>

**tag**
    </dt>
    <dd>

[**```const tensor_convert_to_tag<TDestination>&```**](tensor_convert_to_tag.md)
    </dd>
</dl>

The tag containing the destination underlying type.

### Returns

**```std::shared_ptr<glasssix::tensor<TDestination>>```**

The destination tensor.

<br>

## template<typename TSource, tensor_layout layout><br>operator|(const tensor\<TSource\>&, const tensor_convert_layout_to_tag\<layout\>&)

Converts the layout of a **```glasssix::tensor<T>```** .

```C++
template<typename TSource, tensor_layout layout>
inline tensor<TSource> operator|(const tensor<TSource>& source, const tensor_convert_layout_to_tag<layout>& tag);
```

### Template Parameters.

**```TSource```**

The source underlying type.

<dl>
    <dt>

**layout**
    </dt>
    <dd>

**```tensor_layout```**
    </dd>
</dl>

The destination layout.

### Parameters.

<dl>
    <dt>

**source**
    </dt>
    <dd>

**```const glasssix::tensor<TSource>&```**
    </dd>
</dl>

The source tensor.

<dl>
    <dt>

**tag**
    </dt>
    <dd>

[**```const tensor_convert_layout_to_tag<layout>&```**](tensor_convert_layout_to_tag.md)
    </dd>
</dl>

The tag containing the destination layout.

### Returns

**```glasssix::tensor<TDestination>```**

The destination tensor.

<br>


## template<typename TSource, tensor_layout layout><br>operator|(const std::shared_ptr\<tensor\<TSource\>\>&, const tensor_convert_layout_to_tag\<layout\>&)

Converts the layout of a **```glasssix::tensor<T>```** .

```C++
template<typename TSource, tensor_layout layout>
inline std::shared_ptr<tensor<TSource>> operator|(const std::shared_ptr<tensor<TSource>>& source, const tensor_convert_layout_to_tag<layout>& tag)
```

### Template Parameters

**```TSource```**

The source underlying type.

<dl>
    <dt>

**layout**
    </dt>
    <dd>

**```tensor_layout```**
    </dd>
</dl>

The destination layout.

### Parameters.

<dl>
    <dt>

**source**
    </dt>
    <dd>

**```std::shared_ptr<glasssix::tensor<TSource>>&```**
    </dd>
</dl>

The source tensor.

<dl>
    <dt>

**tag**
    </dt>
    <dd>

[**```const tensor_convert_layout_to_tag<layout>&```**](tensor_convert_layout_to_tag.md)
    </dd>
</dl>

The tag containing the destination tag.

### Returns.

**```std::shared_ptr<glasssix::tensor<TDestination>>```**

The destination tensor.

<br>

## Links
<a href="../Tensor_IO_Manual_zh_CN.md"><img src="../images/home.png" width="32" height="32"></img></a>
