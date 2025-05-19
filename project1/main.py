# import
import os
from pid.pidcontroller import PIDController
from datetime import datetime, timedelta  # don't forget to import datetime
import time
import atexit
import torch
import torchvision
import cv2
from ctrl.base_ctrl import BaseController
import threading

import argparse
args = argparse.ArgumentParser()
args.add_argument('--stream',
        action = 'store_true',
        default= False,
        help = 'Launch OpenCV window and show livestream')
args = args.parse_args()
stream = args.stream


# car = NvidiaRacecar()
# import pygame  # 
from camera import Camera
from ultralytics import YOLO


# Behavior Macro
KEEPING_WAYPOINT = 0
TASK1 = 1 # traffic light
TASK2_STOP = 2 # When the pedestrian is detected
TASK2_SLOW = 3 # ACC
TASK3 = 4 # avoiding obstacles
TASK4 = 5 # straight, left, right

# YOLO Classes Macro
GO_LEFT = 0.0
GO_RIGHT = 1.0
GO_STRAIGHT = 2.0
SIGN_SLOW = 3.0
SIGN_STOP = 4.0
TRAFFIC_GREEN = 5.0
TRAFFIC_RED = 6.0
VEHICLE = 7.0

# Behavior flag
is_go_left = False
is_go_right = False
is_go_straight = False
is_sign_slow = False
is_sign_stop = False
is_traffic_green = False
is_traffic_red = False
is_vehicle = False

# confidence threshold
# TODO: need to change
CONF_THRES = 0.5

# driving constants
MAX_STEER = 0.8
MAX_SPEED = 0.5
STEP_STEER = 0.2
STEP_SPEED = 0.05

# stop
# def stop_driving():
#     car.steering = 0.0
#     car.throttle = 0.0
#     cam.cap[0].release()
# atexit.register(stop_driving) # 터미널 시그널로 종료시 수행



import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS
device = torch.device('cuda')

def preprocess(image: PIL.Image):
    device = torch.device('cuda')
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]



cam = Camera(sensor_id=0, width=640, height=320)
model_yolo = YOLO("ckpts/250519_n_detection.engine")
classes = model_yolo.names
model_alexnet = torchvision.models.alexnet(num_classes=2, dropout=0.0)
model_alexnet.load_state_dict(torch.load('ckpts/lane_best.pt'))
model_alexnet = model_alexnet.to(device)

base = BaseController('/dev/ttyUSB0', 115200)

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def clip(val, max_val):
    return max(min(val, max_val), -max_val)

def update_vehicle_motion(steering, speed):
    steer_val = clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_STEER)

    base_speed = abs(speed_val)

    left_ratio = 1.0 - 1.2*steer_val
    right_ratio = 1.0 + 1.2*steer_val

   # if left_ratio < 0:
   #     left_ratio = 0
   # elif right_ratio < 0:
   #     right_ratio = 0

    L = base_speed * left_ratio
    R = base_speed * right_ratio

    L = clip(L, MAX_SPEED)
    R = clip(R, MAX_SPEED)

    if speed < 0:
        L, R = -L, -R

    send_control_async(-L, -R)
    print(f"[UGV] Speed: {speed_val:.2f}, Steering: {steer_val:.2f} → L: {L:.2f}, R: {R:.2f}")



# def infer_waypoint():
#     image_pil, image_bgr, width, height = get_frame()
#     with torch.no_grad():
#         output = model(preprocess(image_pil)).detach().cpu().numpy()[0]  # (x,y)
#         x = (output[0] / 2 + 0.5) * width
#         y = (output[1] / 2 + 0.5) * height
#     # 시각화 (ESC 종료)
#     # cv2.circle(image_bgr, (int(x), int(y)), 5, (0,0,255), -1)
#     # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     # cv2.imshow("Lane Prediction", image_rgb)
#     if cv2.waitKey(1) == 27:
#         raise KeyboardInterrupt
#     return x, y, width

# def get_frame():

# def preprocess():


# Initialize PID controller
# Using Ziegler-Nichols method
Tu = 0.75
Ku = 2.5
Kp_s = 2.0 * Tu
Ki_s = 0.0
Kd_s = 0.0
Kp_d = 1.5 # Kp_d = 0.6 * Ku    # opt value: 1.5
Ki_d = 3.0 # Ki_d = 1.2 * Ku / Tu    # opt value: 3.0
Kd_d = 0.1875 # Kd_d = 0.075 * Ku * Tu    # opt value: 0.1875
alpha = 0.5     # weight for steering
beta = 0.5      # weight for throttle
throttle_limits = [0.184,0.182]  # Operating area: 전진 0.16, 후진 0.178

# # For headless mode
# os.environ["SDL_VIDEODRIVER"] = "dummy"



running = True # While 문 flag
# capture = False # 사진 저장 flag

frame_counter = 9 # object detection count 변수 -> 10프레임 당 한번만 detect
FRAME_INTERVAL = 10
# bus_detect_time = 0 
# cross_detect_time = None
# stop_time = None
# direction_cls = None
# cls = None
# box_width = 0

pid = PIDController(Kp=Kp_d, Ki=Ki_d, Kd=Kd_d, setpoint=0, steering_limits=(-1.0, 1.0), throttle_limits=throttle_limits)

while running:
# perception

    # 이미지 받아들이기   
    t0 = time.time()
    # timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    _, frame = cam.cap[0].read()

    frame_counter += 1
    if frame_counter % FRAME_INTERVAL == 0:
        ret, frame = cam.cap[0].read()
        if not ret:
            break

        # =============================================
        # yolo = detect
        result = model_yolo(frame)
        print(result[0])
        classes = result[0].boxes.cls.to("cpu").tolist()
        boxes = result[0].boxes.xywh.to("cpu").tolist()
        scores = result[0].boxes.conf.to("cpu").tolist()
        detected_image = result[0].plot()
        # =============================================
        # TODO:
        # change flag -- according to yolo result maybe conf, bbox
        # set threshold needed CONF_THRESH

        # =============================================
        # 시각화 (ESC 종료)
        # if cv2.waitKey(1) == 27:
        #     raise KeyboardInterrupt
        
        # cv2.imshow("Detection", detected_image)
        # cv2.waitKey(1)

        
    # =============================================
    # alexnet = waypoints
    height, width, _ = frame.shape
    frame_pil = PIL.Image.fromarray(frame)
    with torch.no_grad():
        output = model_alexnet(preprocess(frame_pil)).detach().cpu().numpy()[0]  # (x,y)
        x = (output[0] / 2 + 0.5) * width
        y = (output[1] / 2 + 0.5) * height
    print(x, y)

    cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), -1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Lane Prediction", frame)
    cv2.waitKey(1)

    # =============================================
    # if 1. lane following - use x, y
    # image (960, 540) w , h
    center_x = 400 # TODO: CENTER ASSUME!!!!!
    lateral_error = (x - center_x)
    pid.update(output = lateral_error)
    steering_cmd = pid.steering

    # set speed
    forward_speed = 0.18

    update_vehicle_motion(steering_cmd, forward_speed)
    # else 2. cases - use flag classes, boxes
        # generate pid inputs


    # pid calculation


    

        
    # 방향 모드 선택
    # if cls == 'straight' or cls == 'left' or cls == 'right':
    #     direction_cls = cls
    # # elif cls == 'bus' or cls == 'crosswalk':
    #     direction_cls = None

    # get flag


    # switch case

    # normal

    # 
