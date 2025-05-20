# 왼쪽이 steering command 1 방향



import os
# from pid.pidcontroller import SteeringPIDController
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
TASK2 = 2 # When the sign is detected
# TASK2_SLOW = 3 # ACC
TASK3 = 3 # avoiding obstacles
TASK4 = 4 # straight, left, right
mode = KEEPING_WAYPOINT


countTask0 = 0
countTask1 = 0
countTask2 = 0
countTask3 = 0
countTask4 = 0



# YOLO Classes Macro
GO_LEFT = 0.0
GO_RIGHT = 1.0
GO_STRAIGHT = 2.0
SIGN_SLOW = 3.0
SIGN_STOP = 4.0
TRAFFIC_GREEN = 5.0
TRAFFIC_RED = 6.0
VEHICLE = 7.0

# confidence threshold
# TODO: need to change
CONF_THRES = 0.5

# driving constants
MAX_STEER = 3
MAX_SPEED = 0.7
STEP_STEER = 0.2
STEP_SPEED = 0.05
MINIMUM_MOTOR_SPEED = 0.0
MAX_MOTORINPUT = 0.7
# bbox size macro

SIZE_TRAFFIC_LIGHT = 1500.0
SIZE_SLOW_SIGN = 7500.0
SIZE_STOP_SIGN = 7500.0
SIZE_DIRECTION_SIGN = 7500.0



# Behavior flag
is_go_left = False
is_go_right = False
is_go_straight = False
is_sign_slow = False
is_sign_stop = False
is_traffic_green = False
is_traffic_red = False
is_vehicle = False
keep_mode = False

# for gradual stop at traffic light
is_stopping = False
stopping_speed = 0.0
stopping_steer = 0.0

# for gradaul stop at stop sign
is_sign_stopping = False
sign_stopping_speed = 0.0
sign_stopping_steer = 0.0
sign_stop_start_time = None
sign_stop_cooldown = 7.0  # 정지 후 재감지 무시 시간 (초)
last_stop_end_time = 0.0  # 정지 종료 시간 저장용 전역 변수
start_time_task3 = 0.0
keep_mode_end_time = 0.0

ema_alpha = 0.2  # 0~1 사이, 클수록 최신값 반영 큼
use_ema_average = False

avoid_state = {
    "phase": 0,  # 0: 초기, 1: 회피 중, 2: 복귀 중, 3: 종료
    "start_time": None,
    "avoid_dir": 0  # -1: 좌, 1: 우
}

def reset_detection_flags():
    global is_go_left, is_go_right, is_go_straight
    global is_sign_slow, is_sign_stop
    global is_traffic_green, is_traffic_red
    global is_vehicle

    is_go_left = False
    is_go_right = False
    is_go_straight = False
    is_sign_slow = False
    is_sign_stop = False
    is_traffic_green = False
    is_traffic_red = False
    is_vehicle = False



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
# model_alexnet.load_state_dict(torch.load('ckpts/lane_best.pt'))
model_alexnet.load_state_dict(torch.load('ckpts/250520_epoch124.pt'))
model_alexnet = model_alexnet.to(device)

base = BaseController('/dev/ttyUSB0', 115200)

# stop
def stop_driving(steer, speed):
    if abs(speed) < 0.03:
        return 0.0, 0.0
    return steer * 0.8, speed * 0.8

    # cam.cap[0].release()
# atexit.register(stop_driving) # 터미널 시그널로 종료시 수행


def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def clip(val, max_val):
    return max(min(val, max_val), -max_val)

def update_vehicle_motion(steering, speed, mode=None):
    steer_val = clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_SPEED)

    base_speed = abs(speed_val)


    left_ratio = 1.0 - steer_val
    right_ratio = 1.0 + steer_val

   # if left_ratio < 0:
   #     left_ratio = 0
   # elif right_ratio < 0:
   #     right_ratio = 0

    L = base_speed * left_ratio + MINIMUM_MOTOR_SPEED
    R = base_speed * right_ratio + MINIMUM_MOTOR_SPEED

    L = clip(L, MAX_SPEED)
    R = clip(R, MAX_SPEED)

    if speed < 0:
        L, R = -L, -R

    send_control_async(-L, -R)

    print_debug_info(mode, steer_val, speed_val, L, R)

def update_vehicle_motion_hoyan(steering, speed, mode=None):

    motor_minimum_input = 0.4

    # 일단 무조건 양수라고 가정!!!!!!!!!

    steer_val = clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_SPEED)

    base_speed = abs(speed_val)


    left_ratio = 1.0 - 1.2 * steer_val
    right_ratio = 1.0 + 1.2 * steer_val

    # left_ratio = 1.0 - steer_val
    # right_ratio = 1.0 + steer_val

   # if left_ratio < 0:
   #     left_ratio = 0
   # elif right_ratio < 0:
   #     right_ratio = 0

    L = base_speed * left_ratio + MINIMUM_MOTOR_SPEED
    R = base_speed * right_ratio + MINIMUM_MOTOR_SPEED

    L = clip(L, MAX_MOTORINPUT)
    R = clip(R, MAX_MOTORINPUT)

    if speed < 0:
        L, R = -L, -R

    send_control_async(-L, -R)

    print_debug_info(mode, steer_val, speed_val, L, R)


def avoid_obstacles(is_vehicle_visible, vehicle_x, frame_width=960):

    # phase 0: 회피 방향 결정
    if avoid_state["phase"] == 0:
        avoid_state["avoid_dir"] = -1 if vehicle_x > frame_width / 2 else 1
        avoid_state["phase"] = 1
        print(f"회피 시작: {'좌측' if avoid_state['avoid_dir']==-1 else '우측'} 방향으로 회피")

    # phase 1: 장애물이 안보일 때까지 계속 회피 주행
    if avoid_state["phase"] == 1:
        if not is_vehicle_visible:
            avoid_state["start_time"] = time.time()
            avoid_state["phase"] = 2
            print("복귀 주행 시작")
        return avoid_state["avoid_dir"], 0.15  # 우회 시 조향, 감속

    # phase 2: 복귀 주행 (2초)
    if avoid_state["phase"] == 2:
        if time.time() - avoid_state["start_time"] < 2.0:
            return 0.6 * avoid_state["avoid_dir"], 0.15  # 반대 조향, 감속
        else:
            avoid_state["phase"] = 3
            print("원래 주행 모드 복귀")

    # phase 3: 주행 복귀

    if avoid_state["phase"] == 3:
        # flag를 바꾸고
        mode = KEEPING_WAYPOINT
    return -0.3 * avoid_state["avoid_dir"], 0.15

def print_debug_info(mode, steering_cmd, forward_speed, L, R):
    mode_str = {
        KEEPING_WAYPOINT: "KEEPING_WAYPOINT",
        TASK1: "TASK1 (TRAFFIC LIGHT)",
        TASK2: "TASK2 (SIGN : STOP/SLOW)",
        TASK3: "TASK3 (AVOID CAR)",
        TASK4: "TASK4 (DIRECTION SIGN)"
    }.get(mode, "UNKNOWN")

    # 관련 flag만 추려서 출력
    flag_info = ""
    if mode == TASK1:
        flag_info = f"RED: {is_traffic_red}, GREEN: {is_traffic_green}"
    elif mode == TASK2:
        flag_info = f"STOP: {is_sign_stop}, SLOW: {is_sign_slow}"
    elif mode == TASK3:
        flag_info = f"VEHICLE: {is_vehicle}"
    elif mode == TASK4:
        flag_info = f"LEFT: {is_go_left}, RIGHT: {is_go_right}, STRAIGHT: {is_go_straight}"

    print(f"\n[MODE: {mode_str}]    {flag_info}")
    print(f"\tTime :{(time.time() - start_time):.2f}, [Steering: {steering_cmd:.5f}, Speed: {forward_speed:.2f} → L: {L:.2f}, R: {R:.2f} | {flag_info}")
    print(f"countTask0 : {countTask0}, countTask1 : {countTask1}, countTask2 : {countTask2}, countTask3 : {countTask3}, countTask4 : {countTask4}")
def cal_box_size(target_class, result_xywh, result_cls):
    for cls, box in zip(result_cls, result_xywh):
        if cls == target_class:
            _, _, w, h = box  # xywh 포맷
            return w * h
    return None

# Initialize PID controller
# Using Ziegler-Nichols method
Tu = 0.75
Ku = 2.5
Kp_s = 2.0 * Tu
Ki_s = 0.0
Kd_s = 0.0
Kp_d = 0.005 # Kp_d = 0.6 * Ku    # opt value: 1.5
Ki_d = 0.0 # Ki_d = 1.2 * Ku / Tu    # opt value: 3.0
Kd_d = 0.0 # Kd_d = 0.075 * Ku * Tu    # opt value: 0.1875
alpha = 0.5     # weight for steering
beta = 0.5      # weight for throttle
throttle_limits = [0.184,0.182]  # Operating area: 전진 0.16, 후진 0.178

pid = PIDController(Kp=Kp_d, Ki=Ki_d, Kd=Kd_d, setpoint=0, steering_limits=(-1.0, 1.0), throttle_limits=throttle_limits)
# # For headless mode
# os.environ["SDL_VIDEODRIVER"] = "dummy"


# Kp = 0.3
# Ki = 0.0001
# Kd = 0.0001
# pid_controller = SteeringPIDController(Kp, Ki, Kd)


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

start_time = time.time()

while running:
# perception

    # 이미지 받아들이기   
    t0 = time.time()
    # timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    ret, frame = cam.cap[0].read()
    if not ret: continue # 카메라랑 동시에 맞도록...
    
    frame_counter += 1
    if frame_counter % FRAME_INTERVAL == 0:
        ret, frame = cam.cap[0].read()
        if not ret:
            break

        # =============================================
        # yolo = detect
        result = model_yolo(frame)
        classes = result[0].boxes.cls.to("cpu").tolist()
        boxes = result[0].boxes.xywh.to("cpu").tolist()
        scores = result[0].boxes.conf.to("cpu").tolist()
        # =============================================
        # TODO:
        # change flag & mode -- according to yolo result maybe conf, bbox
        # set threshold needed CONF_THRESH
        # after stop_sign change flag in 3s 

        reset_detection_flags()
        next_mode = KEEPING_WAYPOINT
        for cls, box, score in zip(classes, boxes, scores):
            print(f"Class: {cls}, Box: {box}, Score: {score}")

            if cls == GO_LEFT and score > 0.5:
                is_go_left = True
                next_mode = TASK4
            elif cls == GO_RIGHT and score > 0.5:
                is_go_right = True
                next_mode = TASK4
            elif cls == GO_STRAIGHT and score > 0.5:
                is_go_straight = True
                next_mode = TASK4
            elif cls == SIGN_SLOW and score > 0.5:
                is_sign_slow = True
                next_mode = TASK2
            elif cls == SIGN_STOP and score > 0.5:
                is_sign_stop = True
                next_mode = TASK2
            elif cls == TRAFFIC_GREEN and score > 0.5:
                #TODO: bbox size - make def
                is_traffic_green = True
                if (time.time() - start_time) < 45.0:   # traffic 두번 나옴(task1 & 4)
                    next_mode = TASK1
                else:
                    next_mode = TASK4
            elif cls == TRAFFIC_RED and score > 0.5:
                is_traffic_red = True
                if (time.time() - start_time) < 45.0:   # traffic 두번 나옴(task1 & 4)
                    next_mode = TASK1
                else:
                    next_mode = TASK4
            elif cls == VEHICLE and score > 0.5:
                is_vehicle = True
                next_mode = TASK3

        if time.time() > keep_mode_end_time:
            keep_mode = False
        if not keep_mode:
            mode = next_mode
        # =============================================
        # 시각화 (ESC 종료)        if stream:
        # detected_image = result[0].plot()
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
    # print(f"\n\n")
    print(f"x: {x}, y: {y}")

    cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), -1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # cv2.imshow("Lane Prediction", frame)
    # cv2.waitKey(1)

    # =============================================
    # if 1. lane following - use x, y
    # image (960, 540) w , h
    center_x = 420 # TODO: CENTER ASSUME!!!!!
    lateral_error = (x - center_x)

    # for straight forward
    if abs(lateral_error) < 12.0:
        lateral_error = 0.0

    # steering_cmd = pid_controller.update(lateral_error)

    pid.update(output = lateral_error)

    
    steering_cmd = pid.steering
    default_forward_speed = 0.18
    forward_speed = default_forward_speed

    # update_vehicle_motion(steering_cmd, -forward_speed)
    # else 2. cases - use flag classes, boxes
        # generate pid inputs


    # pid calculation

    # # 예시: 현재 모드 변수 (임시로 TASK1로 지정)
    # mode = KEEPING_WAYPOINT

    if mode == KEEPING_WAYPOINT:
        countTask0 += 1
        # 1. 기본 주행 (차선 유지)
        # x, y, lateral_error, pid 등 사용
        steering_cmd = steering_cmd
        forward_speed = default_forward_speed
        
    elif mode == TASK1:
        countTask1 += 1

        # 2. 신호등 인식 및 정지/출발
        traffic_green_size = cal_box_size(TRAFFIC_GREEN, boxes, classes)
        traffic_red_size = cal_box_size(TRAFFIC_RED, boxes, classes)
        print(f"\ttraffic green size: {traffic_green_size}")
        print(f"\ttraffic red size: {traffic_red_size}")
        if is_traffic_red and traffic_red_size > SIZE_TRAFFIC_LIGHT:
            if not is_stopping:
                is_stopping = True
                stopping_speed = forward_speed
                stopping_steer = steering_cmd

            # 점진적으로 감속
            stopping_steer, stopping_speed = stop_driving(stopping_steer, stopping_speed)
            steering_cmd = stopping_steer
            forward_speed = stopping_speed

            # 완전히 멈추면 상태 유지
            if abs(stopping_speed) < 0.03:
                steering_cmd = 0.0
                forward_speed = 0.0

        elif is_traffic_green and traffic_green_size > SIZE_TRAFFIC_LIGHT:
            is_stopping = False
            stopping_speed = 0.0
            steering_cmd = steering_cmd
            forward_speed = 0.18  # 정상 주행 속도로 복귀
    elif mode == TASK2:
        
        countTask2 += 1
    #     # 3. 표지판 인식 정지
        if is_sign_stop:
            # 쿨다운 시간 지나지 않았으면 멈춤 무시
            if time.time() - last_stop_end_time < sign_stop_cooldown:
                # 그냥 정상 주행 유지
                forward_speed = 0.18
            else:
                if not is_sign_stopping:
                    is_sign_stopping = True
                    sign_stopping_speed = forward_speed
                    sign_stopping_steer = steering_cmd
                    sign_stop_start_time = None

                sign_stopping_steer, sign_stopping_speed = stop_driving(sign_stopping_steer, sign_stopping_speed)
                steering_cmd = sign_stopping_steer
                forward_speed = sign_stopping_speed

                if abs(sign_stopping_speed) < 0.03:
                    steering_cmd = 0.0
                    forward_speed = 0.0
                    # 정지 시작 시간 기록 (처음 한 번만)
                    if sign_stop_start_time is None:
                        sign_stop_start_time = time.time()
                    # 정지 유지 시간 (2초 정지 후 다시 주행)    
                    if time.time() - sign_stop_start_time > 2.0:
                        is_sign_stopping = False
                        last_stop_end_time = time.time()  # 정지 종료 시각 기록
                        forward_speed = 0.18  # 다시 주행 속도 복구
        elif is_sign_slow:
            # TODO: make sure to check bbox size
            box_size = cal_box_size(SIGN_SLOW, boxes, classes)
            print(f"\tbox size: {box_size}")
            if(box_size > 7500.0):
                steering_cmd = steering_cmd
                forward_speed = forward_speed * 0.8
        
    elif mode == TASK3:
        
        countTask3 += 1

        keep_mode = True
        keep_mode_end_time = time.time() + 6
        # 처음 실행 시 시작 시간 기록
        if start_time_task3 is None:
            start_time_task3 = time.time()

        elapsed_time = time.time() - start_time_task3

        if elapsed_time >= 2:
            # 2초 이후부터 장애물 회피 로직 실행
            if is_vehicle:
                for cls, box, score in zip(classes, boxes, scores):
                    if cls == VEHICLE and score > 0.5:
                        vehicle_x = box[0]

                steering_cmd, forward_speed = avoid_obstacles(is_vehicle, vehicle_x)

    elif mode == TASK4:
        
        countTask4 += 1
    #     # 6. 방향 전환 (좌/우/직진)
        dir_left_box_size = cal_box_size(GO_LEFT, boxes, classes)
        dir_right_box_size = cal_box_size(GO_RIGHT, boxes, classes)
        dir_straight_box_size = cal_box_size(GO_STRAIGHT, boxes, classes)
        if is_go_left and dir_left_box_size > SIZE_DIRECTION_SIGN:
    #         # 좌회전 로직
            steering_cmd = 1.0
            forward_speed = forward_speed
        elif is_go_right and dir_right_box_size > SIZE_DIRECTION_SIGN:
    #         # 우회전 로직
            steering_cmd = -1.0
            forward_speed = forward_speed
        elif is_go_straight and dir_straight_box_size > SIZE_DIRECTION_SIGN:
            # TODO: test needed 
            steering_cmd = 0.0
            forward_speed = forward_speed


    if(use_ema_average):
        # exponential moving average
        ema_steering = None
        if ema_steering is None: # 최초
            ema_steering = steering_cmd
        else: # 이후
            ema_steering = ema_alpha * steering_cmd + (1 - ema_alpha) * ema_steering
        steering_cmd = ema_steering

    update_vehicle_motion_hoyan(steering_cmd, -forward_speed, mode)

