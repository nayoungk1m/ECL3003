import cv2
import numpy as np
from ultralytics import YOLO
import time
from stereo_depth_estimation.StereoCameraCalibrate.Stereo_Calib_Camera import getStereoCameraParameters, getStereoSingleCameraParameters
import signal
import sys
import Camera.jetsonCam as jetCam 
import os
# from pid.pidcontroller import SteeringPIDController
from pid.pidcontroller import PID_LAT, PID_LONG
from datetime import datetime, timedelta  # don't forget to import datetime
import time
import atexit
import torch
import torchvision
from ctrl.base_ctrl import BaseController
import threading
from datetime import datetime


# ================ constants ================ #

# FRAME_INTERVAL = 10
FRAME_INTERVAL = 3
running = True
frame_counter = -1

# YOLO classes
NO_TRAFFIC_LIGHT = 0.0
TRAFFIC_GREEN = 1.0
TRAFFIC_RED = 2.0
VEHICLE = 3.0  # 차량
SIZE_TRAFFIC_LIGHT = 0.0  # 신호등 크기 기준
class_dict = {3.0: "vehicle", 2.0: "traffic_red", 1.0: "traffic_green", 0.0: "no_traffic_light"}


# detection flags
is_traffic_red = False
is_traffic_green = False
is_vehicle = False

ebs = False  # Emergency Brake System

# SCC constants
current_speed = 0
prev_time = time.time()
imu_prev_time = time.time()

# driving constants
MAX_STEER = 1.2
MAX_SPEED = 0.7
STEP_STEER = 0.2
STEP_SPEED = 0.05
MAX_MOTORINPUT = 0.7
MOTOR_LR_CORRECTION = 1.04
default_forward_speed = 0.19
linear_acceleration = 0.0
forward_speed = 0.0

MINIMUN_FORWARD_SPEED = 0.05
# pid constants

# Initialize PID controller
# Using Ziegler-Nichols method
Tu = 0.75
Ku = 2.5
Kp_s = 0.005
Ki_s = 0.0
Kd_s = 0.0

Kp_d = 0.8
Ki_d = 0.0001
Kd_d = 0.0
# alpha = 0.5     # weight for steering
# beta = 0.5      # weight for throttle
throttle_limits = [0.0, 0.5]  # Operating area: 전진 0.16, 후진 0.178
lateral_pid = PID_LAT(Kp=Kp_s, Ki=Ki_s, Kd=Kd_s, setpoint=0, steering_limits=(-1.0, 1.0))
longitude_pid = PID_LONG(Kp=Kp_d, Ki=Ki_d, Kd=Kd_d, setpoint=0, throttle_limits=throttle_limits)



default_distance = 0.2
imu_ax_offset = None  # IMU ax offset
DEBUG_DRIVEMODE = "normal"  # "defualt", "ACC", "STOP"
# ================ DEBUG ================ #

# 쓸꺼면 수정 필요
codestarttime = datetime.now()
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)  # 디렉토리 없으면 생성

log_filename = f"log_{codestarttime.month:02}_{codestarttime.day:02}_{codestarttime.strftime('%H%M%S')}.txt"
log_path = os.path.join(log_dir, log_filename)

classes, boxes, scores = None, None, None  # 초기화

def debug_print_save(mode, steering_cmd, forward_speed, L, R):

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   

    elapsed_time = time.time() - start_time

    # log_text = (
    #     f"\n[{timestamp}] [MODE: {mode_str}]    {flag_info}\n"
    #     f"\tTime :{elapsed_time:.2f}, [X : {x}], [Steering: {steering_cmd:.5f}, Speed: {forward_speed:.2f} → L: {L:.2f}, R: {R:.2f} | {flag_info}]\n"
    #     f"\t emergency_mode: {emergency_mode}\n"
    #     f"avoid_dir: {avoid_state['avoid_dir']}, phase: {avoid_state['phase']}, change_dir_time: {avoid_state['change_dir_time']}\n"

    #     # f"countTask0 : {countTask0}, countTask1 : {countTask1}, countTask2 : {countTask2}, countTask3 : {countTask3}, countTask4 : {countTask4}\n"
    # )

    # with open(log_path, 'a') as f:
    #     f.write(log_text)
    # print(log_text)

# ================ Video Writer for Debugging ================ #
video_save_dir = "./log"
os.makedirs(video_save_dir, exist_ok=True)
video_filename = datetime.now().strftime("%m_%d_%H%M%S") + ".avi"
video_path = os.path.join(video_save_dir, video_filename)
video_writer = None
video_fps = 7  # 원하는 FPS
video_size = (960, 540)  # hconcat 후 resize 크기


# ================ SCC functions ================ #

def cal_speed(linear_acceleration, dt, forward_speed):
    if(forward_speed < 0.03):
        return 0.0
    return current_speed + linear_acceleration*dt
# ================ control functions ================ #

# stop
def stop_driving(steer, speed):
    # if abs(speed) < 0.03:
    #     return 0.0, 0.0
    # return steer * 0.5, speed * 0.5
    return 0.0, 0.0
    # cam.cap[0].release()
# atexit.register(stop_driving) # 터미널 시그널로 종료시 수행

base = BaseController('/dev/ttyUSB0', 115200)

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def clip(val, max_val):
    return max(min(val, max_val), -max_val)

def update_vehicle_motion(steering, speed, mode=None):

    # 일단 무조건 양수라고 가정!!!!!!!!!

    steer_val = clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_SPEED)

    base_speed = abs(speed_val)

    left_ratio = 1.0 - 1.2 * steer_val
    right_ratio = 1.0 + 1.2 * steer_val

    L = base_speed * left_ratio
    R = base_speed * right_ratio


    L = clip(L, MAX_MOTORINPUT)
    R = clip(R, MAX_MOTORINPUT)

    if speed < 0:
        L, R = -L, -R


    # left right 보정, ratio
    send_control_async(-L * MOTOR_LR_CORRECTION, -R)

    # debug_print_save(mode, steer_val, speed_val, -L, -R)
    
    return -L, -R

# ================ bbox size for traffic light ================ #

def cal_box_size(target_class, result_xywh, result_cls):
    for cls, box in zip(result_cls, result_xywh):
        if cls == target_class:
            _, _, w, h = box  # xywh 포맷
            return w * h
    return None

# ================ reset ================ #

def reset_detection_flags():
    global is_traffic_green, is_traffic_red
    global is_vehicle
    global ebs

    is_traffic_green = False
    is_traffic_red = False
    is_vehicle = False
    ebs = False

# ================ Cameras ================ #

# depth estimation

def signal_handler(sig, frame):
    print('\nExiting...')
    cv2.destroyAllWindows()
    cam1.stop()
    cam2.stop()
    cam1.release()
    cam2.release()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

cam1 = jetCam.jetsonCam()
cam2 = jetCam.jetsonCam()

#left camera 
cam1.open(sensor_id=1,
          sensor_mode=3,
          flip_method=0,
          display_height=540,
          display_width=960,
        )
#right camera
cam2.open(sensor_id=0,
          sensor_mode=3,
          flip_method=0,
          display_height=540,
          display_width=960,
        )

cam1.start()
cam2.start()

lod_data = getStereoCameraParameters('stereo_depth_estimation/jetson_stereo_8MP.npz')
lod_datac1 = getStereoSingleCameraParameters('stereo_depth_estimation/jetson_stereo_8MPc1.npz')
lod_datac2 = getStereoSingleCameraParameters('stereo_depth_estimation/jetson_stereo_8MPc2.npz')

print(lod_datac1[0])
print(lod_datac2[0])
print(lod_datac1[1])
print(lod_datac2[1])

camera_matrix_left = lod_data[0]
dist_coeffs_left =  lod_data[1]
camera_matrix_right =  lod_data[2]
dist_coeffs_right =  lod_data[3]
R =  lod_data[4]
T =  lod_data[5]
# print camera matrix
print("RAW camera matrix")
print(camera_matrix_right)
print(camera_matrix_left)

# 카메라 파라미터 설정
f_x = camera_matrix_left[0, 0]  # 왼쪽 카메라 초점 거리
f_y = camera_matrix_left[1, 1]  # 왼쪽 카메라 초점 거리
c_x = camera_matrix_left[0, 2]  # 왼쪽 카메라 주점 x 좌표
c_y = camera_matrix_left[1, 2]  # 왼쪽 카메라 주점 y 좌표
# 두 카메라 간의 기준 거리(B) (임의 값 설정)
B = 0.4  # 기준 거리, 예시로 10cm 설정 (두 카메라 사이의 거리)

def calculate_depth(disparity):
    """
    깊이 계산 함수
    :param disparity: disparity map
    :return: depth map
    """
    # 0으로 나누는 것을 방지하기 위해 작은 값 추가
    # 깊이 계산
    depth = (f_x * B) / (disparity + 1e-5)
    return depth

def calculate_bbox_depth(disparity, bbox):
    """
    바운딩 박스 내의 평균 깊이 계산 함수
    :param disparity: disparity map
    :param bbox: 바운딩 박스 (x, y, width, height)
    :return: 평균 깊이
    """
    x, y, w, h = map(int, bbox)
    box_disparity = disparity[y - h //8 :y + h//8, x - w//8:x + w//8]
    box_disparity = box_disparity[box_disparity > 0] # -16 제거
    average_depth = np.mean(calculate_depth(box_disparity))
    return average_depth

def undistort_bbox(bbox_xywh):
    x, y, w, h = bbox_xywh
    # 네 꼭짓점 배열 (N,1,2) 형태
    pts = np.array([
        [x - w/2, y - h/2],
        [x + w/2, y - h/2],
        [x + w/2, y + h/2],
        [x - w/2, y + h/2]
    ], dtype=np.float32).reshape(-1,1,2)

    # 왜곡 제거 + 정렬 → P1 좌표계
    pts_rect = cv2.undistortPoints(
        pts,
        camera_matrix_left, dist_coeffs_left,
        R=R1, P=P1      # ← stereoRectify 에서 받은 행렬
    ).reshape(-1,2)

    # 새 박스 좌표 계산
    x_r, y_r = pts_rect[:,0].min(), pts_rect[:,1].min()
    w_r = pts_rect[:,0].max() - x_r
    h_r = pts_rect[:,1].max() - y_r
    return [int(x_r + w_r/2), int(y_r + h_r/2), int(w_r), int(h_r)]

ret,img_s = cam1.read()
if ret:
    image_size = (img_s.shape[1],img_s.shape[0])
    print(image_size)
  
else:
    cam1.stop()
    cam2.stop()
    cam1.release()
    cam2.release()
    exit()

#stereo rectify
R1, R2, P1,P2, Q, roi1, roi2= cv2.stereoRectify(camera_matrix_left,dist_coeffs_left, camera_matrix_right, dist_coeffs_right, image_size, R, T)

block_s = 17
num_disp= 64

# Create a StereoBM object
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_s)

# Load rectification maps
map1_left, map2_left = cv2.initUndistortRectifyMap( camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)




# yolo - object detection

model_yolo = YOLO("models/250615_n_detection.engine", verbose=False)
start_time = time.time()

# alexnet - lane following

import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS
device = torch.device('cuda')

def preprocess(image: PIL.Image):
    device = torch.device('cuda')
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]

model_alexnet = torchvision.models.alexnet(num_classes=2, dropout=0.0)
model_alexnet.load_state_dict(torch.load('models/250607_alexnet_best.pt'))
model_alexnet = model_alexnet.to(device)

model_resnet = torchvision.models.resnet18(num_classes=2)
model_resnet.load_state_dict(torch.load('models/250607_resnet18_last.pt'))
model_resnet = model_resnet.to(device)

while running:
    # ================ perception ================ #
    # Read stereo images
    ret_left, image_left = cam1.read()
    ret_right, image_right = cam2.read()
    if not ret_left: continue # 카메라랑 동시에 맞도록...
    if not ret_right: continue
    frame_counter += 1

    # if frame_counter % 30 == 0:
    #     print(f"Frame: {frame_counter}, Time: {time.time() - start_time:.4f} sec")

    if frame_counter % FRAME_INTERVAL == 0:
        # stereo depth estimation

        # # Read stereo images
        # ret, image_left = cam1.read()
        # ret, image_right = cam2.read()
        # if not ret: continue # 카메라랑 동시에 맞도록...
        # depth_estimation_start_time = time.time()
        # Remap the images using rectification maps
        rectified_left = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(image_right, map1_right, map2_right, cv2.INTER_LINEAR)

        # Convert images to grayscale
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

        # Compute disparity map
        disparity = stereo.compute(gray_left, gray_right)

        normalized_disparity_map = cv2.normalize(disparity, None, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32F)

        colormap_image = cv2.applyColorMap(np.uint8(normalized_disparity_map * 255), cv2.COLORMAP_JET)
        # print(f"Depth estimation time: {time.time() - depth_estimation_start_time} sec")

    
    # if frame_counter % FRAME_INTERVAL == 0:
        # =============================================
        # yolo = detect
        # yolo_detect_start_time = time.time()
        # result = model_yolo(image_left, verbose=False)
        result = model_yolo(image_right, verbose=False)
        classes = result[0].boxes.cls.to("cpu").tolist()
        boxes = result[0].boxes.xywh.to("cpu").tolist()
        scores = result[0].boxes.conf.to("cpu").tolist()
        # print(f"YOLO Detect time: {time.time() - yolo_detect_start_time} sec")

        reset_detection_flags()
        for cls, box, score in zip(classes, boxes, scores):
            if cls == VEHICLE and score > 0.6:

                # Calculate bbox depth
                box_rect = undistort_bbox(box) 
                x, y, w, h = map(int, box_rect)
                roi_percentage = 0.15
                if x <= 960 * roi_percentage or x >= 960 * (1 - roi_percentage):
                    continue

                is_vehicle = True

                average_depth = calculate_bbox_depth(disparity, box_rect)
                print(f"Average depth: {average_depth:.2f} meters")

                if w*h >= 120000.0: # and score > 0.7:
                    # average_depth = 0.1  # 너무 큰 박스는 무시
                    ebs = True
                # Draw bounding box on the colormap image
                cv2.rectangle(colormap_image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)

                cv2.putText(colormap_image, f"Depth: {average_depth:.2f} m", (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            
            if cls == TRAFFIC_RED and score > 0.5:
                is_traffic_red = True
                is_traffic_green = False
                
            if cls == TRAFFIC_GREEN and score > 0.5:
                is_traffic_green = True
                is_traffic_red = False

        # detected_image = result[0].plot()
        # if not isinstance(detected_image, np.ndarray):
        #     detected_image = np.array(detected_image)
    
    
        # com_img=  cv2.hconcat([detected_image, colormap_image])
        # com_img = cv2.resize(com_img, (0, 0), fx=0.6, fy=0.6)

        # # ======== 디버깅 정보 오버레이 ========
        # # Alexnet inference 점 (빨간색)
        # cv2.circle(com_img, (int(x), int(y)), 5, (0, 0, 255), -1)

        # # 우측 하단 텍스트
        # debug_texts = [
        #     f"MODE: {DEBUG_DRIVEMODE}",
        #     f"Avg Depth: {average_depth:.2f} m" if 'average_depth' in locals() else "Avg Depth: N/A",
        #     f"Target Dist: {target_distance:.2f} m",
        #     f"Speed: {forward_speed:.2f}",
        #     f"Steer: {steering_cmd:.2f}",
        #     f"L: {L:.2f}, R: {R:.2f}"
        # ]
        # for i, txt in enumerate(debug_texts):
        #     cv2.putText(com_img, txt, (20, com_img.shape[0] - 20 - 25 * (len(debug_texts) - i - 1)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # # ======== 영상 저장 (매 3프레임) ========
        # if frame_counter % 3 == 0:
        #     if video_writer is None:
        #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #         video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, video_size)
        #     video_writer.write(com_img)

        # cv2.imshow('YOLO - depth map', com_img)
        # cv2.waitKey(1)

    # alexnet = waypoints
    #
    height, width, _ = image_right.shape
    frame_pil = PIL.Image.fromarray(image_right)
    with torch.no_grad():
        lane_time = time.time()
        output = model_alexnet(preprocess(frame_pil)).detach().cpu().numpy()[0]  # (x,y)
        # output = model_resnet(preprocess(frame_pil)).detach().cpu().numpy()[0]
        # print(f"Alexnet inference time: {time.time() - lane_time} sec")
        x = (output[0] / 2 + 0.5) * width
        y = (output[1] / 2 + 0.5) * height
    # print(f"\n\n")
    # print(f"x: {x}, y: {y}")

    cv2.circle(image_right, (int(x), int(y)), 5, (0,0,255), -1)
    image_rgb = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
    # out.write(frame)

    # ================ IMU ================ #
    # imu_data = base.load_imu()
    # if imu_data is not None:

    #     if imu_ax_offset is None:
    #         imu_ax_offset = imu_data.get('ax', 0)

    #     imu_ax = imu_data.get('ax', 0) - imu_ax_offset
    #     linear_acceleration = 0.3 *  imu_ax / 100.0 + 0.7 * linear_acceleration
    #     print(f"IMU ax: {linear_acceleration:.2f} m/s^2")
    #     imu_dt = time.time() - imu_prev_time
 
    #     print(f"dt: {imu_dt:.2f} sec")
    #     imu_prev_time = time.time()

    # print(f"IMU linear acceleration: {linear_acceleration:.2f} m/s^2")
    # 나누기 100 해야됨

    # current_speed = cal_speed(linear_acceleration, imu_dt, forward_speed)
    # ================ planning & control ================ #
    target_distance = default_distance + 0.1 + forward_speed*0.3  # 1.0초 거리 뒤를 따라감
    forward_speed = default_forward_speed
    steering_cmd = 0.0
  
    # 앞차 따라가기(IMU, average_depth 사용)
    # TODO: use & make SCC
    # use linear acceleration of imu
    # use average_depth of depth estimation

    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    # current_speed = cal_speed(linear_acceleration, dt, forward_speed)


    # print(f"Error: {error:.2f} m")

    # ================= pid controller - lateral control ================= 
    center_x = 400 # TODO: CENTER ASSUME!!!!!
    lateral_error = (x - center_x)
    if abs(lateral_error) < 6.0:
        lateral_error = 0.0

    lateral_pid.update_lat(error = lateral_error, dt=dt)
    steering_cmd = lateral_pid.steering
    
    # =================  pid controller - longitudinal control ================= 
    if  is_vehicle and not np.isnan(average_depth): # (average_depth is not None) and 
        longitude_error = average_depth - target_distance
        longitude_pid.update_long(error = longitude_error, dt=dt)
        forward_speed = longitude_pid.throttle + MINIMUN_FORWARD_SPEED
        DEBUG_DRIVEMODE = "ACC"
    else:
        forward_speed = default_forward_speed
        DEBUG_DRIVEMODE = "default"

    # print(f"Steering: {steering_cmd:.2f}, Speed: {forward_speed:.2f}")

    # for red light stop
    # 2. 신호등 인식 및 정지/출발
    traffic_green_size = cal_box_size(TRAFFIC_GREEN, boxes, classes)
    traffic_red_size = cal_box_size(TRAFFIC_RED, boxes, classes)
    if is_traffic_red and traffic_red_size > SIZE_TRAFFIC_LIGHT:
        # is_stopping = True
        forward_speed = 0.0
        steering_cmd = 0.0
        DEBUG_DRIVEMODE = "STOP"

    # print(f"\ttraffic green size: {traffic_green_size}")
    # print(f"\ttraffic red size: {traffic_red_size}")

    # elif is_traffic_green and traffic_green_size > (SIZE_TRAFFIC_LIGHT - 100.0):
        # forward_speed = default_forward_speed  # 정상 주행 속도로 복귀



    if ebs:
        forward_speed = 0.0
        DEBUG_DRIVEMODE = "EBS"
    if forward_speed < 0.0:
        forward_speed = 0.0
    # send control to vehicle
    L, R = update_vehicle_motion(steering_cmd, -forward_speed)


    # ================ visualization ================ #


    detected_image = result[0].plot()
    if not isinstance(detected_image, np.ndarray):
        detected_image = np.array(detected_image)


    com_img=  cv2.hconcat([detected_image, colormap_image])
    com_img = cv2.resize(com_img, (0, 0), fx=0.6, fy=0.6)

    # ======== 디버깅 정보 오버레이 ========

    # 우측 하단 텍스트
    debug_texts = [
        f"MODE: {DEBUG_DRIVEMODE}",
        f"Avg Depth: {average_depth:.2f} m" if 'average_depth' in locals() else "Avg Depth: N/A",
        f"Target Dist: {target_distance:.2f} m",
        # f"Current Speed: {current_speed:.2f}",
        f"Forward Speed: {forward_speed:.2f}",
        f"Steer: {steering_cmd:.2f}",
        f"L: {L:.2f}, R: {R:.2f}"
    ]
    # ACC 모드일 때 PID 값 추가
    if DEBUG_DRIVEMODE == "ACC":
        debug_texts.append(
            f"PID(P:{longitude_pid.p_debug:.2f}, I:{longitude_pid.i_debug:.2f}, D:{longitude_pid.d_debug:.2f})"
        )


    for i, txt in enumerate(debug_texts):
        cv2.putText(image_right, txt, (20, image_right.shape[0] - 20 - 25 * (len(debug_texts) - i - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
    for cls, box, score in zip(classes, boxes, scores):
        # Calculate bbox depth
        box_rect = undistort_bbox(box) 
        x, y, w, h = map(int, box_rect)
        box_size = w * h
        if score > 0.6:
            cv2.rectangle(image_right, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
            # Draw bounding box for depth estimation
            cv2.rectangle(image_right, (x - w // 8, y - h // 8), (x + w // 8, y + h // 8), (255, 0, 0), 2)


        # 클래스명 + box size 표시
        label = f"{class_dict[cls]} ({box_size}) ({score:.2f})"
        cv2.putText(image_right, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ======== 영상 저장 (매 3프레임) ========
    if frame_counter % 3 == 0:
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, video_size)
        if not isinstance(image_right, np.ndarray):
            image_right = np.array(image_right)
        print(image_right.shape)
        video_writer.write(image_right)

    # cv2.imshow('YOLO - depth map', com_img)
    # cv2.waitKey(1)

    # cv2.imshow("Lane Prediction", image_right)
    # cv2.waitKey(1)
    

if video_writer is not None:
    video_writer.release()

