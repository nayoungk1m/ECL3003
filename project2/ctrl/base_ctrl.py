import serial
import json
import queue
import threading
import yaml
import os
import time
import glob
import numpy as np

curpath = os.path.realpath(__file__)
thisPath = os.path.dirname(curpath)
with open(thisPath + '/config.yaml', 'r') as yaml_file:
    f = yaml.safe_load(yaml_file)

class ReadLine:
	def __init__(self, s):
		self.buf = bytearray()
		self.s = s
		self.ANGLE_PER_FRAME = 12
		self.HEADER = 0x54
		self.last_start_angle = 0
		self.breath_light_flag = True

	def readline(self):
		i = self.buf.find(b"\n")
		if i >= 0:
			r = self.buf[:i+1]
			self.buf = self.buf[i+1:]
			return r
		while True:
			i = max(1, min(512, self.s.in_waiting))
			data = self.s.read(i)
			i = data.find(b"\n")
			if i >= 0:
				r = self.buf + data[:i+1]
				self.buf[0:] = data[i+1:]
				return r
			else:
				self.buf.extend(data)

	def clear_buffer(self):
		self.s.reset_input_buffer()

	def read_sensor_data(self):
		if self.sensor_data_ser == None:
			return

		try:
			buffer_clear = False
			while self.sensor_data_ser.in_waiting > 0:
				buffer_clear = True
				sensor_readline = self.sensor_data_ser.readline()
				if len(sensor_readline) <= self.sensor_data_max_len:
					self.sensor_list.append(sensor_readline.decode('utf-8')[:-2])
				else:
					self.sensor_list.append(sensor_readline.decode('utf-8')[:self.sensor_data_max_len])
					self.sensor_list.append(sensor_readline.decode('utf-8')[self.sensor_data_max_len:-2])
			if buffer_clear:
				self.sensor_data = self.sensor_list.copy()
				self.sensor_list.clear()
				self.sensor_data_ser.reset_input_buffer()
		except Exception as e:
			print(f"[base_ctrl.read_sensor_data] error: {e}")


class BaseController:

	def __init__(self, uart_dev_set, buad_set):
		self.ser = serial.Serial(uart_dev_set, buad_set, timeout=1)
		self.rl = ReadLine(self.ser)
		self.command_queue = queue.Queue()
		self.command_thread = threading.Thread(target=self.process_commands, daemon=True)
		self.command_thread.start()

		self.base_light_status = 0
		self.head_light_status = 0

		self.data_buffer = None
		self.base_data = None

		self.use_lidar = f['base_config']['use_lidar']
		self.extra_sensor = f['base_config']['extra_sensor']


	def feedback_data(self):
		try:
			while self.rl.s.in_waiting > 0:
				self.data_buffer = json.loads(self.rl.readline().decode('utf-8'))
				if 'T' in self.data_buffer:
					self.base_data = self.data_buffer
					self.data_buffer = None
					if self.base_data["T"] == 1003:
						print(self.base_data)
						return self.base_data
			# print(self.base_data["T"])
			self.rl.clear_buffer()
			self.data_buffer = json.loads(self.rl.readline().decode('utf-8'))
			self.base_data = self.data_buffer
			return self.base_data
		except Exception as e:
			self.rl.clear_buffer()
			# print(f"[base_ctrl.feedback_data] error: {e}\nraw data: {self.rl.readline().decode('utf-8')}")


	def on_data_received(self):
		self.ser.reset_input_buffer()
		data_read = json.loads(self.rl.readline().decode('utf-8'))
		return data_read


	def send_command(self, data):
		self.command_queue.put(data)


	def process_commands(self):
		while True:
			data = self.command_queue.get()
			self.ser.write((json.dumps(data) + '\n').encode("utf-8"))


	def base_json_ctrl(self, input_json):
		self.send_command(input_json)


	def gimbal_emergency_stop(self):
		data = {"T":0}
		self.send_command(data)


	def base_speed_ctrl(self, input_left, input_right):
		data = {"T":1,"L":input_left,"R":input_right}
		self.send_command(data)


	def gimbal_ctrl(self, input_x, input_y, input_speed, input_acceleration):
		data = {"T":133,"X":input_x,"Y":input_y,"SPD":input_speed,"ACC":input_acceleration}
		self.send_command(data)


	def gimbal_base_ctrl(self, input_x, input_y, input_speed):
		data = {"T":141,"X":input_x,"Y":input_y,"SPD":input_speed}
		self.send_command(data)


	def base_oled(self, input_line, input_text):
		data = {"T":3,"lineNum":input_line,"Text":input_text}
		self.send_command(data)


	def base_default_oled(self):
		data = {"T":-3}
		self.send_command(data)


	def bus_servo_id_set(self, old_id, new_id):
		# data = {"T":54,"old":old_id,"new":new_id}
		data = {"T":f['cmd_config']['cmd_set_servo_id'],"raw":old_id,"new":new_id}
		self.send_command(data)


	def bus_servo_torque_lock(self, input_id, input_status):
		# data = {"T":55,"id":input_id,"status":input_status}
		data = {"T":f['cmd_config']['cmd_servo_torque'],"id":input_id,"cmd":input_status}
		self.send_command(data)


	def bus_servo_mid_set(self, input_id):
		# data = {"T":58,"id":input_id}
		data = {"T":f['cmd_config']['cmd_set_servo_mid'],"id":input_id}
		self.send_command(data)


	def lights_ctrl(self, pwmA, pwmB):
		data = {"T":132,"IO4":pwmA,"IO5":pwmB}
		self.send_command(data)
		self.base_light_status = pwmA
		self.head_light_status = pwmB


	def base_lights_ctrl(self):
		if self.base_light_status != 0:
			self.base_light_status = 0
		else:
			self.base_light_status = 255
		self.lights_ctrl(self.base_light_status, self.head_light_status)


	def gimbal_dev_close(self):
		self.ser.close()


	def change_breath_light_flag(self, input_cmd):
		self.breath_light_flag = input_cmd


	def breath_light(self, input_time):
		self.change_breath_light_flag(True)
		breath_start_time = time.monotonic()
		while time.monotonic() - breath_start_time < input_time:
			for i in range(0, 128, 10):
				if not self.breath_light_flag:
					self.lights_ctrl(0, 0)
					return
				self.lights_ctrl(i, 128-i)
				time.sleep(0.1)
			for i in range(0, 128, 10):
				if not self.breath_light_flag:
					self.lights_ctrl(0, 0)
					return
				self.lights_ctrl(128-i, i)
				time.sleep(0.1)
		self.lights_ctrl(0, 0)

	def load_imu(self):
		data = {"T": 126}  # IMU 데이터 요청
		imu_start_time = time.time()
		self.send_command(data)
		try:
			while True:
				if self.rl.s.in_waiting > 0:
					self.data_buffer = json.loads(self.rl.readline().decode('utf-8'))
					if 'T' in self.data_buffer:
						self.base_data = self.data_buffer
						self.data_buffer = None

						if self.base_data["T"] == 1002:
							print()
							return self.base_data  # IMU 데이터를 반환

						elif self.base_data["T"] == 126:
							# print("[load_imu] Received T:126, but waiting for T:1002 (IMU data).")
							continue
						else:
							return	None # 다른 응답이 온 경우

		except Exception as e:
			self.rl.clear_buffer()
			print(f"[base_ctrl.load_imu] error: {e}")
			return None
