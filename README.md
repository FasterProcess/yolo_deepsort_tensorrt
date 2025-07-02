# YOLO_DeepSort_Tensorrt

run yolo and deepsort with python-TensorRT. Do int8-PTQ quant for both yolo and deepsort-embedding model and deploy by TensorRT. img normalize via torch-GPU, this show how to mix-use TensorRT and Pytorch-GPU without CUDA Context Error.

`./yolo_deepsort_tensorrt/deep_sort_realtime` is not the official original code. I modified `yolo_deepsort_tensorrt/deep_sort_realtime/embedder/mobilenetv2_embedder_trt.py` and `yolo_deepsort_tensorrt/deep_sort_realtime/deepsort_tracker.py` to make TensorRT enable, but it is still compatible with the official code, and allow you set `embedder="mobilenet_trt"` when init DeepSort object to use tensorrt.

# How to generate Calibrator Dataset

You can run yolo to detect objects, then crop the box area and store it in `./dataset/` in PNG or JPG format. Just keep the original resolution and no other special names are required.

# How to export trt engine

You can view https://github.com/oneflyingfish/yolov11_tensorrt

# Introduction

This project provides the code based on python-TensorRT for YOLOv8 or YOLOv11 (they are fully compatible). Note that all onnx here are single-input single-output models.

* PTQ supported
* pipeline-parallel supported
* mix-use with pytorch [fix cuda context and stream error]

`notic:` if you run this code and find that it leans towards only `person` detection, don't worry. This is very likely because the main code has been handled, as I am very concerned about the `person` detected results. In fact, the code of YOLO is complete and it fully considers all detect-types, you can fine-tune the external code by yourself.

# usage

```python
from yolov11_tensorrt import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

yolo = YOLO(
    trt_engine="model/yolov11m_dynamic.engine",
    confidence_thres=0.8,
    iou_thres=0.5,
    max_batch_size=1,
)

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.3,
    gating_only_position=True,          # set True if target has undergone severe deformation.
    embedder="mobilenet_trt",
    embedder_gpu=True,
    bgr=True,                           # set True if your input img is BGR.
    embedding_engine="model/deepsort_embedding.engine",
)

# then use same as official DeepSort
# read imgs
cap = cv2.VideoCapture("test.mp4")
assert cap.isOpened(), "video is not opened"

while True:
    ret, frame = vr.read()
    if not ret:
        break

    yolo.detect_sync(np.expand_dims(frame, axis=0))
    imgs, detections_dicts = yolo.detect_sync_output(wait=True)

    # Transform YOLO detect format to Deepsort format
    dets=[]
    for detect_type, detections in detections_dicts[0].items():
        dets = dets + [
            [detection.box, detection.score, detection.type_name]
            for detection in detections
        ]

    # do track
    track_results = tracker.update_tracks(dets, frame=imgs[0])


tracker.delete_all_tracks()
yolo.release()
tracker.release()
```

# Environment

refer to [how to build environment in ubuntu](env.md)