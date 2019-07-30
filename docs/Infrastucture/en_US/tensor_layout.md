# tensor_layout Enum

## Information

|   |   |
|---|---|
|Header|tensor_layout.hpp|
|Namespace|**```glasssix::excalibur```**|
|Description|Defines the layout of a **```glasssix::tensor<T>```** .

## Syntax

```C++
enum class tensor_layout
{
    rgb,
    rgba,
    grayscale,
    grayscale_3
};
```

## Constants

|   |   |
|---|---|
|rgb|A triple-channel true color image in RGB format.|
|rgba|A quadruplet-channel true color image in RGBA format with an alpha channel.|
|grayscale|A single-channel grayscale image.|
|grayscale_3|A triple-channel grayscale image of which each pixel has three channels with identical values.|

## [Return](../Tensor_IO_Manual_en_US.md)