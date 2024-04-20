from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import serial

app = Flask(__name__,
            static_url_path="",
            static_folder="web/static",
            template_folder="web/templates")
app.config["SECRET_KEY"] = "firefighter"
socketio = SocketIO(app)

ser = serial.Serial("/dev/cu.usbserial-120", 9600, timeout=10)


FIRE_COLOR = (0, 0, 255)
VEST_COLOR = (137, 87, 163)


def get_rect_horizontal_section(frame_width, x, w):
    # Calculate center coordinates of the object and frame
    object_center_x = x + w // 2
    frame_center_x = frame_width // 2

    # Define thresholds for each section based on the frame width
    section_width = frame_width / 3
    left_threshold = section_width
    right_threshold = 3 * section_width

    if object_center_x < left_threshold:
        return "left"
    elif object_center_x > right_threshold:
        return "right"
    else:
        return "middle"


def get_rect_area(w, h):
    return w * h


def get_bounding_box(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    return int(xmin.iloc[0]), int(ymin.iloc[0]), int(width.iloc[0]), int(height.iloc[0])


lower_purple = np.array([120, 50, 100], dtype=np.uint8)
upper_purple = np.array([150, 255, 255], dtype=np.uint8)
min_vest_area = 1000

fire_cascade = cv2.CascadeClassifier("fire_cascade_model_3.xml")


def capture_frames():
    global camera
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()  # read the camera frame
        frame = cv2.resize(frame, (640, 480))
        width, height, _ = frame.shape

        # Calculate section width (all sections are equal)
        section_width = int(width / 3)

        # Draw section separators
        cv2.line(frame, (section_width, 0), (section_width, height),
                 (0, 255, 0), 2)  # Green line for left section boundary
        cv2.line(frame, (3 * section_width, 0), (3 * section_width, height),
                 (0, 0, 255), 2)  # Red line for right section boundary

        if not success:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        vest_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        vest_cnts = cv2.findContours(vest_mask.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
        vest_cnts = sorted(vest_cnts, key=cv2.contourArea, reverse=True)

        if vest_cnts:
            largest_cnt = vest_cnts[0]
            x, y, w, h = cv2.boundingRect(largest_cnt)
            area = cv2.contourArea(largest_cnt)
            section = get_rect_horizontal_section(width, x, w)

            if area >= min_vest_area:
                cv2.rectangle(frame, (x, y), (x + w, y + h), VEST_COLOR, 2)
                data = {
                    "area": area,
                    "x": x, "y": y,
                    "w": w, "h": h
                }
                # socketio.emit("vest", data)
                # print("Vest: {}".format(data))

                if section == "middle":
                    ser.write(bytes(b"forward"))
                    socketio.emit("vest", "forward")
                elif section == "left":
                    ser.write(bytes(b"left"))
                    socketio.emit("vest", "left")
                elif section == "right":
                    ser.write(bytes(b"right"))
                    socketio.emit("vest", "right")
        else:
            socketio.emit("vest", "No vest in sight")

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # fires = fire_cascade.detectMultiScale(frame, 12, 5)
        # if fires is not None:
        #     fires = sorted(fires, key=lambda x: x[2] * x[3], reverse=True)

        # if fires:
        #     x, y, w, h = fires[0]
        #     x = int(x)
        #     y = int(y)
        #     w = int(w)
        #     h = int(h)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), FIRE_COLOR, 2)
        #     area = w * h
        #     data = {
        #         "area": area,
        #         "x": x, "y": y,
        #         "w": w, "h": h
        #     }
        #     socketio.emit("fire", data)
        #     print("Fire: {}".format(data))
        # else:
        #     socketio.emit("fire", "No fire in sight")

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video-feed")
def video_feed():
    return Response(capture_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8001)
