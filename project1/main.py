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
from datetime import datetime
import numpy as np


import argparse
args = argparse.ArgumentParser()
args.add_argument('--stream',
        action = 'store_true',
        default= False,
        help = 'Launch OpenCV window and show livestream')
args = args.parse_args()
stream = args.stream


codestarttime = datetime.now()
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)  # 디렉토리 없으면 생성

log_filename = f"log_{codestarttime.month:02}_{codestarttime.day:02}_{codestarttime.strftime('%H%M%S')}.txt"
log_path = os.path.join(log_dir, log_filename)



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
# CONF_THRES = 0.5

# driving constants
MAX_STEER = 1.2
MAX_SPEED = 0.7
STEP_STEER = 0.2
STEP_SPEED = 0.05
MINIMUM_MOTOR_SPEED = 0.0
MAX_MOTORINPUT = 0.7
default_forward_speed = 0.19
MOTOR_LR_CORRECTION = 1.04

# bbox size macro  
SIZE_TRAFFIC_LIGHT = 1500.0
SIZE_SLOW_SIGN = 6000.0
SIZE_STOP_SIGN = 6000.0
SIZE_DIRECTION_SIGN = 5500.0 # 6000.0 # 6750.0



# Behavior flag
is_go_left = False
is_go_right = False
is_go_straight = False
is_sign_slow = False
is_sign_stop = False
is_traffic_green = False
is_traffic_red = False
is_vehicle = False
emergency_mode = False
turning_mode = False
is_avoiding = False
direction_locked = False
x = 0
y = 0
is_task3_executed = False
# for gradual stop at traffic light
is_stopping = False
stopping_speed = 0.0
stopping_steer = 0.0

# for gradaul stop at stop sign
is_sign_stopping = False
sign_stopping_speed = 0.0
sign_stopping_steer = 0.0
sign_stop_start_time = None
sign_stop_cooldown = 30.0  # 정지 후 재감지 무시 시간 (초)
last_stop_end_time = 0.0  # 정지 종료 시간 저장용 전역 변수
vehicle_first_saw_time = None
emergency_till_thistime = 0.0
turning_timer = 0.0

# task 진입 방지 flag
task1_done = False
task2_done = False
task3_done = False
task4_done = False

ema_alpha = 0.2  # 0~1 사이, 클수록 최신값 반영 큼
use_ema_average = False

avoid_state = {
    "phase": 0,  # 0: 초기, 1: 회피 중, 2: 복귀 중, 3: 종료
    # "phase0_wait_start_time": None,
    "change_dir_time": None,
    "avoid_dir": 0  # -1: 좌, 1: 우
}

def reset_detection_flags():
    global is_go_left, is_go_right, is_go_straight
    global is_sign_slow, is_sign_stop
    global is_traffic_green, is_traffic_red
    global is_vehicle

    # is_go_left = False
    # is_go_right = False
    # is_go_straight = False
    is_sign_slow = False
    is_sign_stop = False
    is_traffic_green = False
    is_traffic_red = False
    is_vehicle = False

output_path = 'lane_prediction_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
fps = 30  # 프레임 속도
frame_size = (960, 540)  # 프레임 사이즈 (frame.shape[1], frame.shape[0])

out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# ==================================================





# ====================================================


import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS
device = torch.device('cuda')

def preprocess(image: PIL.Image):
    device = torch.device('cuda')
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]



cam = Camera(sensor_id=0, width=640, height=320)
model_yolo = YOLO("models/250521_n_detection.engine")
classes = model_yolo.names
model_alexnet = torchvision.models.alexnet(num_classes=2, dropout=0.0)
# model_alexnet.load_state_dict(torch.load('models/lane_best.pt'))
# best.pt 사용 해라 안하면 죽인다
model_alexnet.load_state_dict(torch.load('models/best.pt'))
model_alexnet = model_alexnet.to(device)

base = BaseController('/dev/ttyUSB0', 115200)

# stop
def stop_driving(steer, speed):
    # if abs(speed) < 0.03:
    #     return 0.0, 0.0
    # return steer * 0.5, speed * 0.5
    return 0.0, 0.0
    # cam.cap[0].release()
# atexit.register(stop_driving) # 터미널 시그널로 종료시 수행


def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def clip(val, max_val):
    return max(min(val, max_val), -max_val)

# def update_vehicle_motion(steering, speed, mode=None):
#     steer_val = clip(steering, MAX_STEER)
#     speed_val = clip(speed, MAX_SPEED)

#     base_speed = abs(speed_val)


#     left_ratio = 1.0 - steer_val
#     right_ratio = 1.0 + steer_val

#    # if left_ratio < 0:
#    #     left_ratio = 0
#    # elif right_ratio < 0:
#    #     right_ratio = 0

#     # L = base_speed * left_ratio + MINIMUM_MOTOR_SPEED
#     # R = base_speed * right_ratio + MINIMUM_MOTOR_SPEED
#     L = base_speed * left_ratio + MINIMUM_MOTOR_SPEED
#     R = base_speed * right_ratio + MINIMUM_MOTOR_SPEED

#     L = clip(L, MAX_SPEED)
#     R = clip(R, MAX_SPEED)

#     # if speed < 0:
#     #     L, R = -L, -R

#     send_control_async(-L, -R)

#     print_debug_info(mode, steer_val, speed_val, L, R)
def update_vehicle_motion(steering, speed, mode=None):

    motor_minimum_input = 0.4

    # 일단 무조건 양수라고 가정!!!!!!!!!

    steer_val = clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_SPEED)

    base_speed = abs(speed_val)

    # if abs(steer_val) > 

    left_ratio = 1.0 - 1.2 * steer_val
    right_ratio = 1.0 + 1.2 * steer_val

    L = base_speed * left_ratio  + MINIMUM_MOTOR_SPEED
    R = base_speed * right_ratio  + MINIMUM_MOTOR_SPEED

    L = clip(L, MAX_MOTORINPUT)
    R = clip(R, MAX_MOTORINPUT)

    if speed < 0:
        L, R = -L, -R
    
    # Mininum motor speed 적용
    # if(L < 0 and L > -motor_minimum_input):
    #     L = -motor_minimum_input
    # if(R < 0 and R > -motor_minimum_input):
    #     R = -motor_minimum_input
    # if(L > 0 and L < motor_minimum_input):
    #     L = motor_minimum_input
    # if(R > 0 and R < motor_minimum_input):
    #     R = motor_minimum_input


    # left right 보정, ratio
    send_control_async(-L * MOTOR_LR_CORRECTION, -R)

    debug_print_save(mode, steer_val, speed_val, -L, -R)
    
    return -L, -R


# def avoid_obstacles(is_vehicle_visible, vehicle_x, frame_width=960):

#     # phase 0: 회피 방향 결정
#     if avoid_state["phase"] == 0:
#         avoid_state["avoid_dir"] = -1 if vehicle_x < frame_width / 2 else 1
#         # if avoid_state["phase0_wait_std_dir"], 0.15  # 반대 조향, 감속        
#         avoid_state["phase"] = 1
#         print(f"회피 시작: {'좌측' if avoid_state['avoid_dir']==-1 else '우측'} 방향으로 회피")

#     # phase 1: 장애물이 안보일 때까지 계속 회피 주행
#     if avoid_state["phase"] == 1 or time.time() - avoid_state["start_time"] < 1.0:
#         if not is_vehicle_visible:
#             avoid_state["start_time"] = time.time()
#             avoid_state["phase"] = 2
#             print("복귀 주행 시작")
#         return avoid_state["avoid_dir"], default_forward_speed  # 우회 시 조향, 감속

#     # phase 2: 복귀 주행 (2초)
#     if avoid_state["phase"] == 2:
#         if time.time() - avoid_state["start_time"] > 2.0:
#             avoid_state["phase"] = 3
#             print("원래 주행 모드 복귀")
#         return -0.6 * avoid_state["avoid_dir"], default_forward_speed  # 반대 조향, 감속

#     # phase 3: 주행 복귀

#     if avoid_state["phase"] == 3:
#         # flag를 바꾸고
#         mode = KEEPING_WAYPOINT
#         return -0.6 * avoid_state["avoid_dir"], default_forward_speed  # 반대 조향, 감속

def debug_print_save(mode, steering_cmd, forward_speed, L, R):

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mode_str = {
        KEEPING_WAYPOINT: "KEEPING_WAYPOINT",
        TASK1: "TASK1 (TRAFFIC LIGHT)",
        TASK2: "TASK2 (SIGN : STOP/SLOW)",
        TASK3: "TASK3 (AVOID CAR)",
        TASK4: "TASK4 (DIRECTION SIGN)"
    }.get(mode, "UNKNOWN")

    # flag 정보
    if mode == TASK1:
        flag_info = f"RED: {is_traffic_red}, GREEN: {is_traffic_green}"
    elif mode == TASK2:
        flag_info = f"STOP: {is_sign_stop}, SLOW: {is_sign_slow}"
    elif mode == TASK3:
        flag_info = f"VEHICLE: {is_vehicle}"
    elif mode == TASK4:
        flag_info = f"LEFT: {is_go_left}, RIGHT: {is_go_right}, STRAIGHT: {is_go_straight}, green: {is_traffic_green}, red: {is_traffic_red}"
    else:
        flag_info = ""

    elapsed_time = time.time() - start_time

    log_text = (
        f"\n[{timestamp}] [MODE: {mode_str}]    {flag_info}\n"
        f"\tTime :{elapsed_time:.2f}, [X : {x}], [Steering: {steering_cmd:.5f}, Speed: {forward_speed:.2f} → L: {L:.2f}, R: {R:.2f} | {flag_info}]\n"
        f"\t emergency_mode: {emergency_mode}\n"
        f"avoid_dir: {avoid_state['avoid_dir']}, phase: {avoid_state['phase']}, change_dir_time: {avoid_state['change_dir_time']}\n"

        # f"countTask0 : {countTask0}, countTask1 : {countTask1}, countTask2 : {countTask2}, countTask3 : {countTask3}, countTask4 : {countTask4}\n"
    )

    with open(log_path, 'a') as f:
        f.write(log_text)
    print(log_text)


def cal_box_size(target_class, result_xywh, result_cls):
    for cls, box in zip(result_cls, result_xywh):
        if cls == target_class:
            _, _, w, h = box  # xywh 포맷
            return w * h
    return None

def cal_box_w_h(target_class, result_xywh, result_cls):
    for cls, box in zip(result_cls, result_xywh):
        if cls == target_class:
            _, _, w, h = box  # xywh 포맷
            return w, h
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

def get_current_time():
    return time.time() - start_time

while running:
# perception

    # 이미지 받아들이기   
    # t0 = time.time()????
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
        yolo_detect_start_time = time.time()
        result = model_yolo(frame)
        classes = result[0].boxes.cls.to("cpu").tolist()
        boxes = result[0].boxes.xywh.to("cpu").tolist()
        scores = result[0].boxes.conf.to("cpu").tolist()
        print(f"YOLO Detect time: {time.time() - yolo_detect_start_time} sec")
        # =============================================
        # TODO:
        # change flag & mode -- according to yolo result maybe conf, bbox
        # set threshold needed CONF_THRESH
        # after stop_sign change flag in 3s 

        reset_detection_flags()
        next_mode = KEEPING_WAYPOINT

        for cls, box, score in zip(classes, boxes, scores):
            print(f"Class: {cls}, Box: {box}, Score: {score}")

            if cls == VEHICLE and score > 0.7 and box[3] > 145.0:  # box[2] * box[3] is width * height
                # print(f"box size: {box[2] * box[3]}")
                is_vehicle = True
                vehicle_x = box[0]  # x 좌표
                next_mode = TASK3
            
            elif cls == TRAFFIC_GREEN and score > 0.5:
                is_traffic_green = True
                # task1 실행하면 다시 안들어가게 조절
                if get_current_time() < 35.0:   # traffic 두번 나옴(task1 & 4)
                # if get_current_time() < 3.0:   # traffic 두번 나옴(task1 & 4)
                    next_mode = TASK1
                else:
                    next_mode = TASK4
            elif cls == TRAFFIC_RED and score > 0.5:
                is_traffic_red = True
                if get_current_time() < 35.0:   # traffic 두번 나옴(task1 & 4)
                # if get_current_time() < 3.0:   # traffic 두번 나옴(task1 & 4)

                    next_mode = TASK1
                else:
                    next_mode = TASK4
            elif cls == SIGN_SLOW and score > 0.6:
                is_sign_slow = True
                next_mode = TASK2
            elif cls == SIGN_STOP and score > 0.6:
                is_sign_stop = True
                next_mode = TASK2
            elif not direction_locked:
                if cls == GO_LEFT and score > 0.75 and box[0] >= 960 * 0.25 and box[0] <= 960 * 0.75:
                    is_go_left = True
                    next_mode = TASK4
                    direction_locked = True
                    task4_done = True
                elif cls == GO_RIGHT and score > 0.75:
                    is_go_right = True
                    next_mode = TASK4
                    direction_locked = True
                    task4_done = True
                elif cls == GO_STRAIGHT and score > 0.75:
                    is_go_straight = True
                    next_mode = TASK4
                    direction_locked = True
                    task4_done = True





        if get_current_time() > emergency_till_thistime:
            emergency_mode = False
            
        if not emergency_mode:
            mode = next_mode
        else:
            mode = TASK3  # emergency mode is TASK3

        if task4_done:
            mode = TASK4
        # =============================================
        # 시각화 (ESC 종료)        if stream:
        detected_image = result[0].plot()
        if not isinstance(detected_image, np.ndarray):
            detected_image = np.array(detected_image)

        # 박스 정보(x, y, w, h) 영상에 표시 및 파일 저장
        with open("xywhy1.txt", "a") as xywh_file:
            for cls, box, score in zip(classes, boxes, scores):
                x, y, w, h = box
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                # 파일에 저장 (한 줄에 값들 공백 구분)
                xywh_file.write(f"{cls} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {x1} {y1} {x2} {y2} {score:.3f}\n")

                # 박스 그리기 (더 진한 파란색)
                cv2.rectangle(detected_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                # 텍스트 배경 사각형 추가 (흰색)
                label = f"x:{int(x)} y:{int(y)} w:{int(w)} h:{int(h)} y1:{y1} y2:{y2}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(detected_image, (x1, y1 - th - 8), (x1 + tw, y1), (255, 255, 255), -1)
                # 텍스트(진한 검정)로 표시
                cv2.putText(detected_image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(detected_image)
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
    # out.write(frame)
    # cv2.imshow("Lane Prediction", frame)
    # cv2.waitKey(1)

    # =============================================
    # if 1. lane following - use x, y
    # image (960, 540) w , h
    center_x = 400 # TODO: CENTER ASSUME!!!!!
    lateral_error = (x - center_x)

    # for straight forward
    # boundary
    if abs(lateral_error) < 6.0:
        lateral_error = 0.0

    # steering_cmd = pid_controller.update(lateral_error)

    pid.update(output = lateral_error)

    
    # 기본적으로 사용할 commmand
    steering_cmd = pid.steering
    forward_speed = default_forward_speed

    # update_vehicle_motion(steering_cmd, -forward_speed)
    # else 2. cases - use flag classes, boxes
        # generate pid inputs


    # pid calculationf

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

        elif is_traffic_green and traffic_green_size > SIZE_TRAFFIC_LIGHT:
            is_stopping = False
            stopping_speed = 0.0
            steering_cmd = steering_cmd
            forward_speed = default_forward_speed  # 정상 주행 속도로 복귀


    elif mode == TASK2:
        # TASK 2 들어오면 task1 진입 방지
        task1_done = True
        countTask2 += 1
        stop_box_size = cal_box_size(SIGN_STOP, boxes, classes)
        slow_box_size = cal_box_size(SIGN_SLOW, boxes, classes)


        # STOP SIGN 일 때
        if is_sign_stop and stop_box_size > SIZE_STOP_SIGN:

            if not is_sign_stopping:
                sign_stop_start_time = time.time()
                is_sign_stopping = True
            forward_speed = 0.0

            if time.time() - sign_stop_start_time > 3.0:
                forward_speed = default_forward_speed  # 다시 주행 속도 복구
                task2_done = True
                    

        # SLOW SIGN 일 때           
        elif is_sign_slow and slow_box_size > SIZE_SLOW_SIGN:
            steering_cmd = steering_cmd
            forward_speed = forward_speed * 0.75
        
    elif mode == TASK3:
        # TASK 3 들어오면 task2 진입 방지
        task2_done = True
        countTask3 += 1
                

            

        if not is_task3_executed:
            is_task3_executed = True
            
            # # 처음 실행 시 시작 시간 기록
            # if vehicle_first_saw_time is None:
            #     vehicle_first_saw_time = get_current_time()
            #     emergency_mode = True
            #     emergency_till_thistime = vehicle_first_saw_time + 6.0
            
            
            # 각 단계별 [steering_cmd, speed, duration]을 2D array로 정의
            

            # 장애물 오른쪽에 있어서 왼쪽으로 회피할 때
            if vehicle_x > 960 / 2:
                maneuver_sequence = [
                    [1.0, default_forward_speed, 1.0],    # 좌회전
                    [0.0, default_forward_speed, 0.8],    # 직진
                    [-1.0, default_forward_speed, 1.0],   # 우회전
                    [0.0, default_forward_speed, 1.4],    # 직진
                    [-0.6, default_forward_speed, 1.7],   # 우회전(조금 덜)
                ]
            # 장애물 왼쪽에 있어서 오른쪽으로 회피할 때
            else:
                maneuver_sequence = [
                    [-1.0, default_forward_speed, 1.0],    # 우회전
                    [0.0, default_forward_speed, 0.8],    # 직진
                    [1.0, default_forward_speed, 1.0],   # 좌회전
                    [0.0, default_forward_speed, 1.5],    # 직진
                    [1.0, default_forward_speed, 0.5],   # 좌회전
                    [0.0, default_forward_speed, 1.5],    # 직진
                    [-0.6, default_forward_speed, 0.4],    # 우회전
                ]

            for steering_cmd, forward_speed, duration in maneuver_sequence:
                start_time = time.time()
                while time.time() - start_time < duration:
                    update_vehicle_motion(steering_cmd, -forward_speed, mode)
                    time.sleep(0.05)


        
        # # if is_vehicle:
        #     # steering_cmd, forward_speed = avoid_obstacles(is_vehicle, vehicle_x)
        # # def avoid_obstacles(is_vehicle_visible, vehicle_x, frame_width=960):

        # # phase 0: 회피 방향 결정
        # if avoid_state["phase"] == 0:
        #     avoid_state["avoid_dir"] = -1 if vehicle_x < 960 / 2 else 1   
        #     avoid_state["phase"] = 1
        #     # avoid_state["start_time"] = get_current_time()
        #     print(f"회피 시작: {'좌측' if avoid_state['avoid_dir']==-1 else '우측'} 방향으로 회피")

        # # phase 1: 장애물이 안보일 때까지 계속 회피 주행
        # if avoid_state["phase"] == 1: # or time.time() - avoid_state["start_time"] < 1.0:
        #     if is_vehicle:
        #         steering_cmd = avoid_state["avoid_dir"] * 0.6
        #         forward_speed = default_forward_speed  # 우회 시 조향, 감속
        #     else:
        #         if avoid_state["change_dir_time"] == None :
        #             avoid_state["change_dir_time"] = get_current_time()
        #             # avoid_state["change_dir_time"] = time.time()

        #         if get_current_time() - avoid_state["change_dir_time"] < 1.0:
        #         # if time.time() - avoid_state["change_dir_time"] < 1.0:
        #             steering_cmd = 0
        #         else:
        #             avoid_state["phase"] = 2
        #             avoid_state["change_dir_time"] = get_current_time()
        #             # print("복귀 주행 시작")
        #     # return avoid_state["avoid_dir"], default_forward_speed  # 우회 시 조향, 감속

        # # phase 2: 복귀 주행 (2초)
        # if avoid_state["phase"] == 2:
        #     if get_current_time() - avoid_state["change_dir_time"] < 1.50:
        #         steering_cmd = - avoid_state["avoid_dir"]
        #         forward_speed = default_forward_speed  # 반대 조향, 감속
        #     else:
        #         avoid_state["phase"] = 3
        #         emergency_till_thistime = get_current_time()
        #         # task3_done = True
        #         print("원래 주행 모드 복귀")
        #     # return -0.6 * avoid_state["avoid_dir"], default_forward_speed  # 반대 조향, 감속

        # # phase 3: 주행 복귀

        # # if avoid_state["phase"] == 3:
        # #     # flag를 바꾸고
        # #     # return -0.6 * avoid_state["avoid_dir"], default_forward_speed  # 반대 조향, 감속
        # #     # steering_cmd = -0.6 * avoid_state["avoid_dir"]
        # #     forward_speed = default_forward_speed  # 반대 조향, 감속


    elif mode == TASK4:
        # TASK 4 들어오면 task3 진입 방지
        task3_done = True
        countTask4 += 1
        # 신호등 판단
        traffic_green_size = cal_box_size(TRAFFIC_GREEN, boxes, classes)
        traffic_red_size = cal_box_size(TRAFFIC_RED, boxes, classes)

        #     # 6. 방향 전환 (좌/우/직진)
        dir_left_box_size = cal_box_size(GO_LEFT, boxes, classes)
        dir_right_box_size = cal_box_size(GO_RIGHT, boxes, classes)
        dir_straight_box_size = cal_box_size(GO_STRAIGHT, boxes, classes)

        if is_traffic_red and traffic_red_size > SIZE_TRAFFIC_LIGHT:
            
            # steering_cmd = 0.0
            forward_speed = 0.0

            
            # if not is_stopping:
            #     is_stopping = True
            #     stopping_speed = forward_speed
            #     stopping_steer = steering_cmd

            # # 점진적으로 감속
            # # stopping_steer, stopping_speed = stop_driving(stopping_steer, stopping_speed)
            # # steering_cmd = stopping_steer
            # # forward_speed = stopping_speed
            # forward_speed = 0.0

            # # 완전히 멈추면 상태 유지
            # if abs(stopping_speed) < 0.03:
            #     steering_cmd = 0.0
            #     forward_speed = 0.0
        # elif is_traffic_green and traffic_green_size > SIZE_TRAFFIC_LIGHT:
        else:
            # 초록불일때 주행 시작
        # else:
            # is_stopping = False
            # stopping_speed = 0.0

            # if turning_mode is not None:
            #     if time.time() - turning_timer < 5.0:
            #         if turning_mode == 'left':
            #             # steering_cmd = 1.2
            #             steering_cmd = MAX_STEER

            #         elif turning_mode == 'right':
            #             steering_cmd = -MAX_STEER

            #             # steering_cmd = -1.2
            #         elif turning_mode == 'straight':
            #             steering_cmd = 0.0
            #     # 속도는 그대로
            #     forward_speed = forward_speed
            # else:
            #     turning_mode = None 
            if is_go_left and dir_left_box_size > SIZE_DIRECTION_SIGN:
            #         # 좌회전 로직
                # steering_cmd = 1.5
                steering_cmd = 1.0
                forward_speed = forward_speed
                # turning_mode = 'left'
                turning_start_time = time.time()
                while time.time() - turning_start_time < 1.0:
                    update_vehicle_motion(steering_cmd, -forward_speed, mode)
                    time.sleep(0.05) 
                while True:
                    steering_cmd = 0.0  # 회전 후 직진으로 변경
                    forward_speed = default_forward_speed
                    update_vehicle_motion(steering_cmd, -forward_speed, mode)
                    time.sleep(0.05) 

            elif is_go_right and dir_right_box_size > SIZE_DIRECTION_SIGN:
            #         # 우회전 로직
                steering_cmd = -1.0
                forward_speed = forward_speed
                # turning_mode = 'right'
                turning_start_time = time.time()
                while time.time() - turning_start_time < 1.0:
                    update_vehicle_motion(steering_cmd, -forward_speed, mode)
                    time.sleep(0.05) 
                while True:
                    steering_cmd = 0.0  # 회전 후 직진으로 변경
                    forward_speed = default_forward_speed
                    update_vehicle_motion(steering_cmd, -forward_speed, mode)
                    time.sleep(0.05) 
            elif is_go_straight and dir_straight_box_size > SIZE_DIRECTION_SIGN:
                steering_cmd = 0.0
                forward_speed = forward_speed
                # turning_mode = 'straight'

                # turning_start_time = time.time()
                # while time.time() - turning_start_time < 1.0:
                #     update_vehicle_motion(steering_cmd, -forward_speed, mode)
                #     time.sleep(0.05) 
                while True:
                    steering_cmd = 0.0  # 회전 후 직진으로 변경
                    forward_speed = default_forward_speed
                    update_vehicle_motion(steering_cmd, -forward_speed, mode)
                    time.sleep(0.05) 


    if(use_ema_average):
        # exponential moving average
        ema_steering = None
        if ema_steering is None: # 최초
            ema_steering = steering_cmd
        else: # 이후
            ema_steering = ema_alpha * steering_cmd + (1 - ema_alpha) * ema_steering
        steering_cmd = ema_steering

    L, R = update_vehicle_motion(steering_cmd, -forward_speed, mode)


