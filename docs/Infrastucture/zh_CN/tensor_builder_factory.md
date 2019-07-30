# tensor_builder_factory 类

## 信息

|   |   |
|---|---|
|头文件|tensor_builder_factory.hpp|
|命名空间|**```glasssix::excalibur```**|
|功能描述|创建一个 [**```tensor_builder```**]() 的示例。

## 语法

```C++
class tensor_builder_factory final
{
public:
    static std::shared_ptr<tensor_builder> create(tensor_builder_implementation type = tensor_builder_implementation::free_image);
};
```

## 成员函数

|   |   |
|---|---|---|
|[**create(tensor_builder_implementation)**]()|创建一个 **```tensor_builder```** 的实例。|

## [返回](../Tensor_IO_Manual_zh_CN.md)