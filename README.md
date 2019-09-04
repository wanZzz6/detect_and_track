# detect_and_track
目标检测与跟踪Demo

从视频流（视频、摄像头）中识别并跟踪目标，画出其运动轨迹）

包含YOLOV3 和SSD 两种模型的使用

SSD 模型最多可检测21种类别的目标

使用说明：

- deep_sort_ssd.py: 使用tensorflow (cpu版) ，效果 18~30 fps（i9处理器）
- deep_sort_yolov3.py 使用tensorflow-gpu，效果：6~8fps（cuda 10 + 2080Ti）
