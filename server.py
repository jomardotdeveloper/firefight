from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send
import cv2
import torch

app = Flask(__name__,
            static_url_path="",
            static_folder="web/static",
            template_folder="web/templates")
app.config['SECRET_KEY'] = 'firefighter'
socketio = SocketIO(app)


object_model = torch.hub.load("ultralytics/yolov5", "yolov5s")
fire_model = torch.hub.load("yolov5", "custom", source="local", path="models/fire_best.pt")
fire_model.conf = 0.2
fire_model.iou = 0.2

FIRE_COLOR = (255, 0, 0)
OBJECT_COLOR = (0, 255, 0)


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


def capture_frames():
    global camera
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            continue

        fire_results = fire_model(frame, size=640)
        for xyxy in fire_results.pandas().xyxy:
            fire_df = xyxy[xyxy["name"] == "fire"]
            if fire_df.empty:
                continue

            try:
                x, y, w, h = get_bounding_box(fire_df["xmin"], fire_df["ymin"], fire_df["xmax"], fire_df["ymax"])
                cv2.rectangle(frame, (x, y), (x+w, y+h), FIRE_COLOR, 2)
                section = get_rect_horizontal_section(frame, x, w)
                area = get_rect_area(w, h)

                fire_message = "FIRE_{section}_{area}".format(section=section, area=area)
                socketio.send(fire_message)
                print(fire_message)
            except:
                continue

        # object_results = object_model(frame, size=640)
        # for xyxy in object_results.pandas().xyxy:
        #     try:
        #         x, y, w, h = get_bounding_box(xyxy["xmin"], xyxy["ymin"], xyxy["xmax"], xyxy["ymax"])
        #         cv2.rectangle(frame, (x, y), (x+w, y+h), OBJECT_COLOR, 2)
        #         section = get_rect_horizontal_section(frame, x, w)
        #         area = get_rect_area(w, h)

        #         object_message = "OBJECT_{section}_{area}".format(section=section, area=area)
        #         socketio.send(object_message)
        #         print(object_message)
        #     except:
        #         continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video-feed")
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=8001)
