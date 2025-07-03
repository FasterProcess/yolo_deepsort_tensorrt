from yolov11_tensorrt import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np



def draw_img(frame,track_results):
    write_img = frame.copy()
    for track_result in track_results:
        if not track_result.is_confirmed():
            continue
        tid = track_result.track_id
        x1, y1, x2, y2 = map(int, track_result.to_ltrb())
        cv2.rectangle(write_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"id: {tid}"
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
    return write_img

def main():
    cap = cv2.VideoCapture("test/example.mp4")
    out_cap = cv2.VideoWriter(
        "test/output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    assert cap.isOpened(), "video is not opened"
    
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
        gating_only_position=True,         
        embedder="mobilenet_trt",
        embedder_gpu=True,
        bgr=True,                           
        embedding_engine="model/deepsort_embedding.engine",
    )   # same usage as official DeepSort


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections_dicts =yolo.detect(np.expand_dims(frame, axis=0))
        dets=[]
        if "person" in detections_dicts[0]:
            detections = detections_dicts[0]["person"]
            dets = [
                [detection.box, detection.score, detection.type_name]
                for detection in detections
            ]
        
        # do track
        track_results = tracker.update_tracks(dets, frame=frame)
        out_cap.write(draw_img(frame,track_results))


    tracker.delete_all_tracks()
    cap.release()
    out_cap.release()
    yolo.release()
    tracker.release()
    
main()