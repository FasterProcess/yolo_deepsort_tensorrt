import sys

sys.path.insert(0, ".")

import os
import csv
import random
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from decord import VideoReader, cpu
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import time
from threading import Thread
from yolov11_tensorrt import YOLO, DetectBox
from queue import Queue
import cv2


def read_video(video_path, use_opencv=False) -> VideoReader | cv2.VideoCapture:
    if use_opencv:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"fail to open {video_path}"
        return cap, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        vr = VideoReader(video_path, ctx=cpu(0))
        return vr, len(vr)


def tracker_sync(
    yolo: YOLO,
    tracker: DeepSort,
    save_detect=False,
):
    video_idx = 0
    idx = 0
    time_cost = 0
    start_track = False
    tracker.push_ctx()
    write_cap = None
    output_path = ""
    while True:
        data = yolo.detect_sync_output(wait=True)
        if data is None:
            break

        if isinstance(data, bool):
            tracker.delete_all_tracks()
            idx = 0
            time_cost = 0
            start_track = False
            video_idx += 1
            if write_cap is not None:
                write_cap.release()
                print(f"write result to {output_path}")
                output_path = ""
                write_cap = None
            continue

        imgs, detections_dicts = data

        if save_detect and write_cap is None:
            os.makedirs(os.path.join("outputs/"), exist_ok=True)
            output_path = os.path.join("outputs/", f"video_{video_idx}.mp4")
            write_cap = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                25,
                (
                    imgs.shape[2],
                    imgs.shape[1],
                ),
            )

        for i in range(len(imgs)):
            write_img = None
            frame, detections_dict = imgs[i], detections_dicts[i]
            if "person" in detections_dict:
                detections = detections_dict["person"]
                raw_dets = [
                    [detection.box, detection.score, detection.type_name]
                    for detection in detections
                ]
                start_track = True
                # if save_detect:
                #     # draw yolo bbox
                #     write_img = frame.copy()
                #     for detection in detections:
                #         x1, y1, w, h = detection.box
                #         cv2.rectangle(
                #             write_img,
                #             (int(x1), int(y1)),
                #             (int(x1 + w), int(y1 + h)),
                #             (255, 0, 0),
                #             3,
                #         )
            else:
                raw_dets = []

            if not start_track:
                if write_img is not None:
                    write_cap.write(write_img)
                continue

            start = time.time()
            track_results = tracker.update_tracks(raw_dets, frame=frame)
            time_cost += time.time() - start
            idx += 1

            tid_cc = -1
            for track_result in track_results:
                tid_cc += 1
                if not track_result.is_confirmed():
                    continue
                tid = track_result.track_id
                x1, y1, x2, y2 = map(int, track_result.to_ltrb())

                if write_img is not None:
                    cv2.rectangle(write_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"id: {tid_cc}"
                    # Calculate the dimensions of the label text
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    label_x = x1
                    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
                    cv2.rectangle(
                        write_img,
                        (label_x, label_y - label_height),
                        (label_x + label_width, label_y + label_height),
                        (0, 0, 255),
                        cv2.FILLED,
                    )

                    # Draw the label text on the image
                    cv2.putText(
                        write_img,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    write_cap.write(write_img)

    tracker.pop_ctx()
    print("tracker thread exit")


def detect_and_save(
    args: argparse.Namespace,
    vr: VideoReader | cv2.VideoCapture,
    yolo: YOLO,
):
    if isinstance(vr, VideoReader):
        skip = min(args.skip, len(vr) // 30)
        for i in range(0, len(vr), skip):
            yolo.detect_sync(vr.get_batch([i]).asnumpy())
    else:
        assert vr.isOpened(), "video is not opened"
        i = -1
        skip = min(args.skip, int(vr.get(cv2.CAP_PROP_FRAME_COUNT)) // 30)
        while True:
            ret, frame = vr.read()
            i += 1
            if not ret:
                break
            elif i % skip == 0:
                i == 0
                yolo.detect_sync(np.expand_dims(frame, axis=0))
    yolo.detect_sync(True)  # send end flag
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_fold",
        type=str,
        default="test/",
        help="fold to store input videos",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="model/yolov11m_dynamic.engine",
        help="Path to YOLO TRT model",
    )
    parser.add_argument(
        "--embedding_engine",
        type=str,
        default="model/deepsort_embedding.engine",
        help="Path to Deepsort Embedding TRT model",
    )
    parser.add_argument(
        "--decord_read",
        type=bool,
        default=False,
        help="use decord to read video, False means opencv",
    )
    parser.add_argument(
        "--save_detect",
        type=bool,
        default=True,
        help="save detect result to debug/",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="frame to skip read",
    )
    args = parser.parse_args()

    yolo = YOLO(
        trt_engine=args.model_path,
        confidence_thres=0.8,
        iou_thres=0.5,
        max_batch_size=1,
    )
    tracker = DeepSort(
        max_age=max(50 // args.skip, 5),
        n_init=max(25 // args.skip, 5),
        max_iou_distance=0.7,
        max_cosine_distance=0.3,
        gating_only_position=True,
        embedder=(
            "mobilenet_trt"
            if args.embedding_engine is not None
            and args.embedding_engine[-7:] == ".engine"
            else "mobilenet"
        ),
        embedder_gpu=True,
        bgr=True,
        embedding_engine=args.embedding_engine,
    )

    tracker_thread = Thread(
        target=tracker_sync,
        args=(yolo, tracker, args.save_detect),
    )
    tracker_thread.start()

    idx = 0
    video_names = os.listdir(args.video_fold)
    for idx, video_name in enumerate(video_names):
        video_path = os.path.join(args.video_fold, video_name)
        vr, _ = read_video(video_path, use_opencv=not args.decord_read)
        print(f"[INFO] Processing {video_name}, video-idx={idx}")
        detect_and_save(
            args,
            vr,
            yolo,
        )

    yolo.release()
    tracker_thread.join()
    tracker.release()


if __name__ == "__main__":
    main()
