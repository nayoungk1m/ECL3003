from base_ctrl import BaseController
import time
from pynput import keyboard
import threading
import os
from datetime import datetime, timedelta  # don't forget to import datetime

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


now = datetime.now()
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)  # 디렉토리 없으면 생성

log_filename = f"log_{now.month:02}_{now.day:02}_{now.strftime('%H%M%S')}.txt"
log_path = os.path.join(log_dir, log_filename)

def save_log(log_path, mode, steering_cmd, forward_speed, L, R):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # mode_str = {
    #     KEEPING_WAYPOINT: "KEEPING_WAYPOINT",
    #     TASK1: "TASK1 (TRAFFIC LIGHT)",
    #     TASK2: "TASK2 (SIGN : STOP/SLOW)",
    #     TASK3: "TASK3 (AVOID CAR)",
    #     TASK4: "TASK4 (DIRECTION SIGN)"
    # }.get(mode, "UNKNOWN")

    # flag 정보
    # if mode == TASK1:
    #     flag_info = f"RED: {is_traffic_red}, GREEN: {is_traffic_green}"
    # elif mode == TASK2:
    #     flag_info = f"STOP: {is_sign_stop}, SLOW: {is_sign_slow}"
    # elif mode == TASK3:
    #     flag_info = f"VEHICLE: {is_vehicle}"
    # elif mode == TASK4:
    #     flag_info = f"LEFT: {is_go_left}, RIGHT: {is_go_right}, STRAIGHT: {is_go_straight}"
    # else:
        # flag_info = ""

    elapsed_time = time.time() - now

    log_text = (
        # f"\n[{timestamp}] [MODE: {mode_str}]    {flag_info}\n"
        f"\tTime :{elapsed_time:.2f}, [Steering: {steering_cmd:.5f}, Speed: {forward_speed:.2f} → L: {L:.2f}, R: {R:.2f} " #| {flag_info}]\n"
        # f"countTask0 : {countTask0}, countTask1 : {countTask1}, countTask2 : {countTask2}, countTask3 : {countTask3}, countTask4 : {countTask4}\n"
    )

    with open(log_path, 'a') as f:
        f.write(log_text)


# ====================================================



try:
    while listener.running:
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
