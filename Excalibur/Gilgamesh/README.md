# Gilgamesh

A wrapper and test project of Tensor Operations in Excalibur.

## Comparison with OpenCV on x86-64

- CPU: Intel i7-6700k
- GPU: GTX TitanX(Maxwell) * 2

### Speed
| Method             |    OpenCV   | Native CPU  | SIMD CPU | GPU |
|--------------------|-------------|-------------|-----------|
|resize              | --          | --          | --  |
|flip             | --      | --       | -- |
|grayscale| --      | --       | -- |
|cut           |--      | --       | -- |
|copymakeborder |--      | --       | -- |
|rotate				 |--      | --       | -- |

