from .base_ctrl import BaseController
import time
from pynput import keyboard
import threading
import os
from datetime import datetime, timedelta  # don't forget to import datetime
from ..camera import Camera
import cv2
import numpy as np

# === Constants ===
MAX_STEER = 2.0
MAX_SPEED = 0.3
STEP_STEER = 0.5
STEP_SPEED = 0.05

base = BaseController('/dev/ttyUSB0', 115200)

steering = 0.0
speed = 0.0

pressed_keys = set()
last_update_time = 0
update_interval = 0.1

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def on_press(key):
    try:
        pressed_keys.add(key.char)
    except:
        pass

def on_release(key):
    try:
        pressed_keys.discard(key.char)
        if key.char == 'q':
            return False  # 종료
    except:
        pass

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

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

now = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'/home/ircv20/project/project1/videos/video_{now}.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 변경
fps = 30  # 프레임 속도
frame_size = (960, 540)  # 프레임 사이즈 (frame.shape[1], frame.shape[0])

out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

cam = Camera(sensor_id=0)

# ====================================================



try:
    while listener.running:
        ret, frame = cam.cap[0].read()

        out.write(frame)

        now = time.time()

        if (now - last_update_time) >= update_interval:
            if 's' in pressed_keys:
                speed += STEP_SPEED
            elif 'w' in pressed_keys:
                speed -= STEP_SPEED
            else:
                speed *= 0.9

            if 'a' in pressed_keys:
                steering += STEP_STEER
            elif 'd' in pressed_keys:
                steering -= STEP_STEER
            else:
                steering *= 0.5

            update_vehicle_motion(steering, speed)
            last_update_time = now

except KeyboardInterrupt:
    print("\n Quit")
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
