# 基于[NCNN](https://github.com/Tencent/ncnn)实现在Android App调用YOLOv5/v7/v8的目标检测/关键点模型,进行推理预测

 支持 [yolov5](https://github.com/ultralytics/yolov5),  [edgeai-yolov5](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose),  [yolov7](https://github.com/WongKinYiu/yolov7),  [yolov8](https://github.com/ultralytics/ultralytics)

 系统: Android 5.0+(21)

[English](README.md)


# 如何搞事情?

1 下载[.aar](https://github.com/wkt/YoloMobile/releases)文件

2 把.aar放到自己Android相关模块的libs目录(例如:app/libs)下

3 编辑app/build.gradle, 添加
```
implementation files('libs/yolo_mobile_release_2023xxyyzz_V1.0r1.aar')
```

4 把ncnn格式模型文件.bin and .param放到assets目录下

5 在assets目录下,创建文件'yolo_cfg.json'
```
{
  "name": "yolov8n",
  "input_size": 384,
  "param": "yolov.param",
  "bin": "yolo.bin",
  "box_thr": 0.5,
  "iou_thr": 0.5,
  "nkpt": 0, 
  "ver": 8,
  "outputs": [
    {"name": "345","stride":8,"anchors": [10,13, 16,30, 33,23]},
    {"name": "365","stride":16,"anchors": [30,61, 62,45, 59,119]},
    {"name": "385","stride":32,"anchors": [116,90, 156,198, 373,326]}
  ],
  
  "names": [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush" ]

}
```
字段说明:

  input_size -- 模型输入图像大小, 目前仅支持w=h, 例如: 640

  input_name -- 模型入口节点名称

  outputs    -- 模型出口节点名称列表

可选字段:

  ver     -- yolo v8需要设为8

  names   -- 类别名称

  nkpt    -- 目标关键点数量, 例如: 17


6 调用模型, 进行推理预测
```

        infer = new YoloInfer(ctx);
        infer.loadFromConfigAssets("yolo_cfg.json");
        ...
        List<YoloInfer.Box> boxes = infer.detect(bitmap);
        ...
        YoloInfer.draw(canvas,boxes,paint);

```

# 演示Demo
 演示Demo的[apk](https://github.com/wkt/YoloMobile/releases/download/v1.0.2r3/app-debug.apk)
 
 截图:
 <img src="images/20230918_214448.png">


# 重建.aar文件
```
bash gradlew :YoloMobile:assembleRelease
```
