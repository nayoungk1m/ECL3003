from ctrl.base_ctrl import BaseController
import time

base = BaseController('/dev/ttyUSB0', 115200)
while True:
    start_time = time.time()
    imu_data = base.load_imu()
    print(f"Processing time: {time.time() - start_time} sec")
    # print(imu_data)
    print(f"ax: {imu_data['ax']}, ay: {imu_data['ay']}, az: {imu_data['az']}")