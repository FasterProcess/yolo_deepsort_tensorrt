# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import torch
import argparse
import pycuda.driver as cuda
from typing import List, Tuple, Dict
import os
import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue
import tensorrt as trt
from .memory import HostDeviceMem
from .detect_box import DetectBox

a = torch.randn(size=(1, 1), device=torch.device("cuda:0"))
del a


class YOLOv8_11Trt:
    """
    YOLOv8_11 object detection model class for handling ONNX inference and visualization.

    This class provides functionality to load a YOLOv8_11 ONNX model, perform inference on images,
    and visualize the detection results with bounding boxes and labels.

    Attributes:
        onnx_model (str): Path to the ONNX model file.
        input_image (str): Path to the input image file.
        confidence_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        classes (List[str]): List of class names from the COCO dataset.
        color_palette (np.ndarray): Random color palette for visualizing different classes.
        input_width (int): Width dimension of the model input.
        input_height (int): Height dimension of the model input.
        img (np.ndarray): The loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.

    Methods:
        letterbox: Resize and reshape images while maintaining aspect ratio by adding padding.
        draw_detections: Draw bounding boxes and labels on the input image based on detected objects.
        preprocess: Preprocess the input image before performing inference.
        postprocess: Perform post-processing on the model's output to extract and visualize detections.
        main: Perform inference using an ONNX model and return the output image with drawn detections.

    Examples:
        Initialize YOLOv8_11 detector and run inference
        >>> detector = YOLOv8_11("YOLOv8_11n.onnx", "image.jpg", 0.5, 0.5)
        >>> output_image = detector.main()
    """

    def __init__(
        self,
        trt_engine: str,
        confidence_thres: float,
        iou_thres: float,
        max_batch_size=-1,
        force_torch_input=True,
    ):
        """
        Initialize an instance of the YOLOv8_11 class.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        self.force_torch_input = force_torch_input
        self.preprocess_input_queue = Queue()
        self.inference_input_queue = Queue()
        self.postprocess_input_queue = Queue()
        self.postprocess_output_queue = Queue()

        self.pre_thread = Thread(target=self.preprocess_thread)
        self.infer_thread = Thread(target=self.inference_thread)
        self.post_thread = Thread(target=self.postprocess_thread)

        # init cuda context
        cuda.init()
        self.gpu = cuda.Device(0)
        self.cuda_ctx = self.gpu.make_context()

        self.trt_engine = trt_engine
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Generate a color palette for the classes
        print(f"load model: {self.trt_engine}")
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.trt_logger)
        with open(self.trt_engine, "rb") as f:
            self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())

        self.trt_context = self.trt_engine.create_execution_context()
        self.cuda_stream = cuda.Stream()

        # Get the model inputs
        self.input_name = self.trt_engine.get_tensor_name(0)
        self.output_name = self.trt_engine.get_tensor_name(1)
        # set input shape
        self.max_input_shape = [
            int(dim)
            for dim in self.trt_engine.get_tensor_profile_shape(self.input_name, 0)[2]
        ]
        # self.max_input_shape = [
        #     int(dim) for dim in self.trt_engine.get_tensor_shape(self.input_name)
        # ]
        assert (
            max_batch_size <= self.max_input_shape[0]
        ), f"your batchsize is bigger than {self.max_input_shape[0]}, consider re-generate engine"

        if max_batch_size > 0:
            self.max_input_shape[0] = max_batch_size
        self.trt_context.set_input_shape(self.input_name, tuple(self.max_input_shape))
        # get output shape
        self.max_input_shape = [
            int(dim) for dim in self.trt_context.get_tensor_shape(self.input_name)
        ]
        self.max_output_shape = [
            int(dim) for dim in self.trt_context.get_tensor_shape(self.output_name)
        ]
        self.input_shape = None
        self.output_shape = None

        # print(f'input: "{self.input_name}", max shape: {self.max_input_shape}')
        # print(f'output: "{self.output_name}", max_shape: {self.max_output_shape}')

        self.input_mem = HostDeviceMem(
            self.max_input_shape, np.float32, self.cuda_stream
        )
        self.output_mem = HostDeviceMem(
            self.max_output_shape, np.float32, self.cuda_stream
        )

        self.set_input_shape(self.max_input_shape)

        self.color_palette = np.random.uniform(0, 255, size=(80, 3))
        self.input_width = self.max_input_shape[-1]
        self.input_height = self.max_input_shape[-2]

        self.warmup()

        self.pre_thread.start()
        self.infer_thread.start()
        self.post_thread.start()

    def set_input_shape(self, shape):
        shape = [int(dim) for dim in shape]
        if shape != self.input_shape:
            self.input_shape = shape
            self.trt_context.set_input_shape(self.input_name, tuple(self.input_shape))
            # get output shape
            self.output_shape = [
                int(dim) for dim in self.trt_context.get_tensor_shape(self.output_name)
            ]

            self.input_mem.set_shape(self.input_shape)
            self.output_mem.set_shape(self.output_shape)

            self.trt_context.set_tensor_address(
                self.input_name, int(self.input_mem.ptr())
            )
            self.trt_context.set_tensor_address(
                self.output_name, int(self.output_mem.ptr())
            )
            # print(
            #     f"reset input shape to input shape: {self.input_shape}, output shape: {self.output_shape}"
            # )

    def detect_sync(self, imgs: np.ndarray):
        self.preprocess_input_queue.put(imgs)

    def detect_sync_output(
        self, wait=False
    ) -> Tuple[np.ndarray, List[Dict[str, List[DetectBox]]]] | bool:
        if self.postprocess_output_queue.empty() and not wait:
            return None
        else:
            return self.postprocess_output_queue.get()

    def preprocess_thread(self):
        while True:
            data = self.preprocess_input_queue.get()
            if data is None:
                self.inference_input_queue.put(None)
                return
            if isinstance(data, bool):
                self.inference_input_queue.put(data)
                continue
            img_datas, pads = self.preprocess(data)
            if self.force_torch_input:
                img_datas = torch.from_numpy(img_datas).cuda().contiguous()
            self.inference_input_queue.put((img_datas, pads, data))

    def inference_thread(self):
        self.cuda_ctx.push()
        while True:
            data = self.inference_input_queue.get()
            if data is None:
                self.postprocess_input_queue.put(None)
                break
            if isinstance(data, bool):
                self.postprocess_input_queue.put(data)
                continue
            img_datas, pads, imgs = data
            output = self.inference(img_datas)
            self.postprocess_input_queue.put((imgs, output, pads))
        self.cuda_ctx.pop()
        # self.cuda_ctx.detach()

    def postprocess_thread(self):
        while True:
            data = self.postprocess_input_queue.get()
            if data is None:
                self.postprocess_output_queue.put(None)
                return
            if isinstance(data, bool):
                self.postprocess_output_queue.put(data)
                continue
            imgs, output, pads = data
            self.postprocess_output_queue.put(
                (imgs, self.postprocess(imgs, output, pads))
            )

    def release(self):
        self.preprocess_input_queue.put(None)
        self.pre_thread.join()
        self.infer_thread.join()
        self.post_thread.join()

        self.cuda_stream.synchronize()
        del self.trt_context
        del self.trt_engine
        del self.trt_runtime
        self.cuda_ctx.detach()
        # del self.cuda_ctx

    def warmup(self, times=3):
        for _ in range(times):
            self.inference(np.ndarray(self.input_shape, dtype=np.float32))

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (
            new_shape[0] - new_unpad[1]
        ) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (top, left)

    def draw_detections(
        self, imgs: np.ndarray, detect_boxses: List[Dict[str, List[DetectBox]]]
    ) -> None:
        """Draw bounding boxes and labels on the input image based on the detected objects."""
        # Extract the coordinates of the bounding box

        for i in range(len(imgs)):
            img = imgs[i]
            detect_boxs = detect_boxses[i]
            for type_name in detect_boxs:
                for detect_box in detect_boxs[type_name]:
                    x1, y1, w, h = detect_box.box
                    # Retrieve the color for the class ID
                    color = self.color_palette[detect_box.type_id]
                    # Draw the bounding box on the image
                    cv2.rectangle(
                        img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2
                    )
                    # Create the label text with class name and score
                    label = f"{detect_box.type_name}: {detect_box.score:.2f}"
                    # Calculate the dimensions of the label text
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    # Calculate the position of the label text
                    label_x = x1
                    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

                    # Draw a filled rectangle as the background for the label text
                    cv2.rectangle(
                        img,
                        (label_x, label_y - label_height),
                        (label_x + label_width, label_y + label_height),
                        color,
                        cv2.FILLED,
                    )

                    # Draw the label text on the image
                    cv2.putText(
                        img,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

    def preprocess(self, imgs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            image_datas (np.ndarray): Preprocessed image data ready for inference with shape (B, 3, height, width).
            pads (np.ndarray): Padding values (B, top, left) applied during letterboxing.
        """
        # Convert the image color space from BGR to RGB
        image_datas = []
        pads = []
        for i in range(len(imgs)):
            img = imgs[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, pad = self.letterbox(img, (self.input_width, self.input_height))
            image_data = np.transpose(img, (2, 0, 1))  # Channel first
            image_data = np.array(image_data) / 255.0
            image_datas.append(image_data)
            pads.append(pad)
        return np.stack(image_datas, dtype=np.float32), np.array(pads, dtype=np.int32)

    def postprocess(
        self, input_image: np.ndarray, outputs: List[np.ndarray], pads: np.ndarray
    ) -> List[Dict[str, List[DetectBox]]]:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs_b = outputs[0]
        batch_size = outputs_b.shape[0]
        outputs_b = np.transpose(outputs_b, axes=(0, 2, 1))
        max_score_b = np.amax(outputs_b[..., 4:], axis=-1)

        img_width, img_height = input_image.shape[-2], input_image.shape[-3]

        result_batch = []
        for b in range(batch_size):
            outputs = outputs_b[b]
            pad = pads[b]
            # Get the number of rows in the outputs array
            rows = outputs.shape[0]

            # Lists to store the bounding boxes, scores, and class IDs of the detections
            boxes = []
            scores = []
            class_ids = []

            # Calculate the scaling factors for the bounding box coordinates
            gain = min(
                self.input_height / img_height,
                self.input_width / img_width,
            )
            outputs[:, 0] -= pad[1]
            outputs[:, 1] -= pad[0]

            # Iterate over each row in the outputs array
            for i in range(rows):
                # Extract the class scores from the current row
                classes_scores = outputs[i][4:]

                # Find the maximum score among the class scores
                # max_score = np.amax(classes_scores)
                max_score = max_score_b[b][i]

                # If the maximum score is above the confidence threshold
                if max_score >= self.confidence_thres:
                    # Get the class ID with the highest score
                    class_id = np.argmax(classes_scores)

                    # Extract the bounding box coordinates from the current row
                    x, y, w, h = (
                        outputs[i][0],
                        outputs[i][1],
                        outputs[i][2],
                        outputs[i][3],
                    )

                    # Calculate the scaled coordinates of the bounding box
                    left = int((x - w / 2) / gain)
                    top = int((y - h / 2) / gain)
                    width = int(w / gain)
                    height = int(h / gain)

                    # Add the class ID, score, and box coordinates to the respective lists
                    x0, y0 = max(0, min(left, img_width)), max(0, min(top, img_height))
                    x1, y1 = max(0, min(left + width, img_width)), max(
                        0, min(top + height, img_height)
                    )

                    left, top, width, height = x0, y0, x1 - x0, y1 - y0

                    if width * height > 0:
                        class_ids.append(class_id)
                        scores.append(max_score)
                        boxes.append([left, top, width, height])

            # Apply non-maximum suppression to filter out overlapping bounding boxes
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.confidence_thres, self.iou_thres
            )

            result = {}
            for i in indices:
                if boxes[i][2] * boxes[i][3] > 0:
                    type_name = DetectBox.classes[class_ids[i]]
                    if type_name not in result:
                        result[type_name] = []
                    result[type_name].append(
                        DetectBox(scores[i], boxes[i], class_ids[i])
                    )

            result_batch.append(result)
        return result_batch

    def inference(
        self, img_datas: np.ndarray | torch.Tensor, require="np"
    ) -> np.ndarray | torch.Tensor:
        self.set_input_shape(img_datas.shape)

        if isinstance(img_datas, np.ndarray):
            self.input_mem.set_numpy(img_datas)
        else:
            self.input_mem.set_torch(img_datas)
        self.trt_context.execute_async_v3(
            stream_handle=self.cuda_stream.handle,
        )

        if require == "np":
            return [self.output_mem.read_numpy()]
        else:
            return [self.output_mem.read_torch()]

    def detect(
        self, imgs: np.ndarray | torch.Tensor
    ) -> List[Dict[str, List[DetectBox]]]:
        """
        Perform inference using an ONNX model and return the output image with drawn detections.

        imgs: (N,H,W,C) float32 ndarray

        Returns:
            detect boxes for eache frame, length is N
        """

        self.detect_sync(imgs=imgs)
        return self.detect_sync_output(wait=True)[1]
