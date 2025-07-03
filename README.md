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

```bash
python3 main.py
```

# Environment

> complete development environment refer to [how to build development environment in ubuntu](env/env.md)

```
pip3 install env/requirements.txt
```