from pathlib import Path
import torch

import cv2
import serial


FIRE_CASCADE_MODEL_PATH = Path(__file__).parent / "fire_cascade_model.xml"

FIRE_COLOR = (255, 0, 0)


def get_rect_horizontal_section(frame, x, w):
    frame_width, _ = frame.shape[1], frame.shape[0]  # Get frame dimensions

    # Calculate center coordinates of the object and frame
    object_center_x = x + w // 2
    frame_center_x = frame_width // 2

    # Define thresholds for each section based on the frame width
    left_threshold = frame_width * 1 / 3
    right_threshold = frame_width * 2 / 3

    if object_center_x < left_threshold:
        return "left"
    elif left_threshold <= object_center_x <= right_threshold:
        return "middle"
    else:
        return "right"


def get_rect_area(w, h):
    return w * h


def get_bounding_box(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    return int(xmin.iloc[0]), int(ymin.iloc[0]), int(width.iloc[0]), int(height.iloc[0])


def main():
    fire_model = torch.hub.load("yolov5", "custom", source="local", path="models/fire_best.pt")
    fire_model.conf = 0.2
    fire_model.iou = 0.2
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        results = fire_model(frame, size=640)

        for xyxy in results.pandas().xyxy:
            fire_df = xyxy[xyxy["name"] == "fire"]
            if fire_df.empty:
                continue
            x, y, w, h = get_bounding_box(fire_df["xmin"], fire_df["ymin"], fire_df["xmax"], fire_df["ymax"])
            cv2.rectangle(frame, (x, y), (x+w, y+h), FIRE_COLOR, 2)
            section = get_rect_horizontal_section(frame, x, w)
            area = get_rect_area(w, h)
            print("FIRE_{section}_{area}".format(section=section, area=area))
            # Serial.write("FIRE_{section}_{area}")

        cv2.imshow("frame", frame)

        key = cv2.waitKey(20)
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
