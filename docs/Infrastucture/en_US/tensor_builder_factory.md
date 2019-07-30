# tensor_builder_factory Class

## Information

|   |   |
|---|---|
|Header|tensor_builder_factory.hpp|
|Namespace|**```glasssix::excalibur```**|
|Description|A class factory to create an instance of [**```tensor_builder```**]() .

## Syntax

```C++
class tensor_builder_factory final
{
public:
    static std::shared_ptr<tensor_builder> create(tensor_builder_implementation type = tensor_builder_implementation::free_image);
};
```

## Member Functions

|   |   |
|---|---|---|
|[**create(tensor_builder_implementation)**]()|Creates an instance of **```tensor_builder```** .|

## [Return](../Tensor_IO_Manual_en_US.md)