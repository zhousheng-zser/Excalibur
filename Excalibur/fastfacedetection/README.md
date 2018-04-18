# fastfacedetection

A fast library for face detection and face landmark detection with high accuracy based on excalibur. The face detection speed can reach 50FPS on GTX TitanX(Maxwell).

## Comparison on x86-64

- CPU: Intel i7-6700k
- CPU Math Library: OpenBLAS 0.2.19
- GPU: GTX TitanX(Maxwell) * 2
- GPU Math Library: CUDA8.0, cuDNN7.1
- min_window_size: 48

### The 1000 average forward time(ms) on CPU(Intel i7-6700k)
|  image resolution  | Caffe |  mini-Caffe  |  Excalibur  |
| :------: | :------:| :------: | :------: |
| 640*480   |  -- |  --  |  115  |
| 1280*720   |  -- |  --  |  171  |
| 1920*1080   | -- |  --  |  247  |

### The 1000 average forward time(ms) on GPU(NVIDIA GTX TitanX) with/without cuDNN
|  image resolution  | Caffe |  mini-Caffe  |  Excalibur  |
| :------: | :------:| :------: | :------: |
| 640*480   |  55.8/ |  --  |  27.9/  |
| 1280*720   |  87.6/ |  --  |  42.1/  |
| 1920*1080   | 251/ |  --  |  63.7/  |

## Comparison on ARM

## Accuracy Preformance
### visible lighting condition


### infrared lighting condition
