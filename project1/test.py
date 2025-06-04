


while time.time() - turning_start_time < 1.25:
    update_vehicle_motion(steering_cmd, -forward_speed, mode)
    time.sleep(0.05) 
go_straight_time = time.time()

while time.time() - go_straight_time < 1.0:
    steering_cmd = 0.0  # 회전 후 직진으로 변경
    forward_speed = default_forward_speed
    update_vehicle_motion(steering_cmd, -forward_speed, mode)
    time.sleep(0.05) 
turning_start_time = time.time()
while time.time() - turning_start_time < 1.25:
    steering_cmd = 0.0  # 회전 후 직진으로 변경
    forward_speed = default_forward_speed
    update_vehicle_motion(steering_cmd, -forward_speed, mode)
    time.sleep(0.05) 