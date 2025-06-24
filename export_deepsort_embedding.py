from deep_sort_realtime.embedder.embedder_pytorch import (
    MobileNetv2_Embedder as Embedder,
)
import torch
import onnx

embedder = Embedder(
    half=False,
    max_batch_size=16,
    bgr=True,
    gpu=False,
    model_wts_path=None,
)

embedding_model = embedder.model
embedding_model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    embedding_model,
    dummy_input,
    "model/deepsort_embedding.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
)

onnx_model = onnx.load("model/deepsort_embedding.onnx")
onnx_model_shape = onnx.shape_inference.infer_shapes(onnx_model)
onnx.save_model(onnx_model_shape, "model/deepsort_embedding_infer_shape.onnx")

import os
os.remove("model/deepsort_embedding.onnx")
os.rename("model/deepsort_embedding_infer_shape.onnx", "model/deepsort_embedding.onnx")
