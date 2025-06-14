from datetime import datetime, timedelta  # don't forget to import datetime
import time
import numpy as np



class PID_LAT:
    def __init__(self, Kp, Ki, Kd, setpoint=0, steering_limits=(-1.0, 1.0)):
 
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0         # cumulative error
        self.pre_error = 0.0        # previous error
        self.pre_t = None           # previous time
        self.setpoint = setpoint    # target setpoint, 가운데가 0이 아닐수도 있음. Offset 조절 필요할지도 // 우린 이미 밖에서 해줌

        self.steering = 0.0
        self.Ka = 2.0/Kp        # Anti-Windup coefficient
        self.anti = 0.0
    
    def update_lat(self, error, stopflag=False, ICflag = False, dt=0.0):
        # output must be "current value from sensor", not the actual output of the function
        # The actual output of the function is "self.steering" & "self.throttle". You should call this variable to use for car.steering & car.throttle
        # error = self.setpoint - output
        error = - error

        # cur_t = datetime.now()  # needs to import `datetime`
        # if self.pre_t is None:
        #     dt = 0.0
        # else:
        #     dt = (cur_t - self.pre_t).total_seconds()
        #     # print(f"dt: {dt}")

        # self.pre_t = cur_t
        derivative = 0.0

        if dt > 0.0:
            # self.integral += error * dt   # without anti-windup
            # if stopflag: # 신호등 확인으로 정지 시
            #     derivative = 0.0
            # elif ICflag: # 교차로 진입 시
            #     self.integral += (error - self.Ka * self.anti) * dt
            #     derivative = 0.0
            # else:
            self.integral += (error - self.Ka * self.anti) * dt
            derivative = (error - self.pre_error) / dt
        else:
            # when dt == 0, prevent dividing zero
            derivative = 0.0

        self.pre_error = error



        self.steering = self.Kp * error # + self.Ki * self.integral + self.Kd * derivative
        

        # Anti-windup: Clamp the integral term if control is beyond limits
        # if (self.steering_limits[0] is not None) and (self.steering < self.steering_limits[0]):
        #     self.steering = self.steering_limits[0]
        #     if error < 0:  # Prevent further negative integral accumulation
        #         self.integral -= error * dt # error * dt < 0, increase integral
        # elif (self.steering_limits[1] is not None) and (self.steering > self.steering_limits[1]):
        #     self.steering = self.steering_limits[1]
        #     if error > 0:  # Prevent further positive integral accumulation
        #         self.integral -= error * dt # error * dt > 0, decrease integral

        # if self.steering > self.steering_limits[1]:
        #     saturated = self.steering_limits[1]
        # elif self.steering < self.steering_limits[0]:
        #     saturated = self.steering_limits[0]
        # else:
        #     saturated = self.steering
        
        saturated = self.steering


        self.anti = self.steering - saturated
        self.steering = saturated
        

class PID_LONG:
    def __init__(self, Kp, Ki, Kd, setpoint=0, throttle_limits=(0.0, 0.25)):
 
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0         # cumulative error
        self.integral_limit = 0.5  # limit for integral term to prevent windup
        self.pre_error = 0.0        # previous error
        self.pre_t = None           # previous time
        self.setpoint = setpoint    # target setpoint, 가운데가 0이 아닐수도 있음. Offset 조절 필요할지도 // 우린 이미 밖에서 해줌
        self.throttle_limits = throttle_limits
        self.throttle = 0.0
        self.Ka = 2.0/Kp        # Anti-Windup coefficient

        self.p_debug = 0.0
        self.i_debug = 0.0
        self.d_debug = 0.0
        
    
    def update_long(self, error, stopflag=False, ICflag = False, dt=0.0):
        # output must be "current value from sensor", not the actual output of the function
        # The actual output of the function is "self.steering" & "self.throttle". You should call this variable to use for car.steering & car.throttle
        # error = self.setpoint - output

        # cur_t = datetime.now()  # needs to import `datetime`
        # if self.pre_t is None:
        #     dt = 0.0
        # else:
        #     dt = (cur_t - self.pre_t).total_seconds()
        #     # print(f"dt: {dt}")

        # self.pre_t = cur_t
        if error is None:
            return

        if dt > 0.0 and not np.isnan(error):
            # self.integral += error * dt   # without anti-windup
            # if stopflag: # 신호등 확인으로 정지 시
            #     derivative = 0.0
            # elif ICflag: # 교차로 진입 시
            #     self.integral += (error - self.Ka * self.anti) * dt
            #     derivative = 0.0
            # else:
            # self.integral += (error - self.Ka * self.anti) * dt
            self.integral += (error) * dt
            derivative = (error - self.pre_error) / dt
        else:
            # when dt == 0, prevent dividing zero
            derivative = 0.0

        self.pre_error = error


        # Anti-windup: Limit the integral term to prevent windup
        if abs(self.integral) > self.integral_limit:
            if self.integral > 0:
                self.integral = self.integral_limit
            else:
                self.integral = -self.integral_limit



        self.throttle = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.p_debug = self.Kp * error
        self.i_debug = self.Ki * self.integral
        self.d_debug = self.Kd * derivative


        print(f"Throttle: {self.throttle}")
        print(f"\tError: {error}, integral: {self.integral}, derivative: {derivative}, dt: {dt}")
        print()

        # Anti-windup: Clamp the integral term if control is beyond limits
        # if (self.steering_limits[0] is not None) and (self.steering < self.steering_limits[0]):
        #     self.steering = self.steering_limits[0]
        #     if error < 0:  # Prevent further negative integral accumulation
        #         self.integral -= error * dt # error * dt < 0, increase integral
        # elif (self.steering_limits[1] is not None) and (self.steering > self.steering_limits[1]):
        #     self.steering = self.steering_limits[1]
        #     if error > 0:  # Prevent further positive integral accumulation
        #         self.integral -= error * dt # error * dt > 0, decrease integral

        # if self.steering > self.steering_limits[1]:
        #     saturated = self.steering_limits[1]
        # elif self.steering < self.steering_limits[0]:
        #     saturated = self.steering_limits[0]
        # else:
        #     saturated = self.steering
        
        # saturated = self.throttle


        # self.anti = self.throttle - saturated
        # self.throttle = saturated
        

        # max_throttle = self.throttle_limits[1]  # Maximum throttle value
        # min_throttle = self.throttle_limits[0]  # Minimum throttle value
        # self.throttle = min_throttle + (max_throttle - min_throttle) * abs(self.steering)