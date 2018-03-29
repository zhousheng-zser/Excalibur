# exfastfacedetection

A very fast library for face detection and face landmark detection in images. The face detection speed can reach 1500FPS on GTX 1080Ti.

## Comparison on x86-64

- CPU: Intel i7-6700k
- GPU: GTX TitanX(Maxwell)
- min_window_size: 48
- npd model: 1223-1

### 640*480 resolution
| Method             |Time         | FPS         |Time         | FPS        | CPU Usage(Aveage) |
|--------------------|-------------|-------------|-------------|------------|-----------|
|                    |   X64       |  X64        |  X64        |X64         |           |
|                    |Single-thread|Single-thread|Multi-thread |Multi-thread|           |
|OpenCV              | --          | --          | 12.33ms     |     81.1   | --  |
|frontal             | 3.41ms      | 414.9       | 0.652ms     | 1533.1     | 17% |
|frontal-surveillance| 4.37ms      | 269.7       | 0.944ms     | 1059.8     | 14% |
|multiview           | 7.81ms      | 172.1       | 1.597ms     |  626.4     | 17% |
|multiview_reinforce | 11.15ms     | 109.3       | 2.725ms     |  367.0     | 16% |
|cunpd				 | 6.72ms      | 184.8       | 0.84ms     | 1190.4     | 13% |

### 1920*1080 resolution
| Method             |Time         | FPS         |Time         | FPS        | CPU Usage(Aveage) |
|--------------------|-------------|-------------|-------------|------------|-----------|
|                    |   X64       |  X64        |  X64        |X64         |           |
|                    |Single-thread|Single-thread|Multi-thread |Multi-thread|           |
|OpenCV              | --          | --          | --     |   --  | --  |
|frontal             | 20.65ms      | 49.1   | --     | --    | 17% |
|frontal-surveillance| 23.11ms      | 43.5      | --    | --     | 16% |
|multiview           | 37.92ms      | 27.0       | --     |  --  | 17% |
|multiview_reinforce | 55.43ms     | 18.0      | -- |  --   | 16% |
|cunpd				 | 15.24ms      | 65.8       | --   | --    | 8% |

## Comparison on ARM

## Accuracy Preformance
### visible lighting condition
![visibal](img/visible_lighting.png)

### infrared lighting condition
![infrared](img/infrared_lighting.jpg)