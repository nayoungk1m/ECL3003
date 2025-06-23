from datetime import datetime, timedelta  # don't forget to import datetime
import time


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, steering_limits=(-1.0, 1.0),throttle_limits=(0.19, 0.20)):
 
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0         # cumulative error
        self.pre_error = 0.0        # previous error
        self.pre_t = None           # previous time
        self.setpoint = setpoint    # target setpoint, 가운데가 0이 아닐수도 있음. Offset 조절 필요할지도 // 우린 이미 밖에서 해줌
        self.steering_limits = steering_limits  # (min_output, max_output)
        self.throttle_limits = throttle_limits
        self.steering = 0.0
        # self.throttle = 0.0
        self.Ka = 2.0/Kp        # Anti-Windup coefficient
        self.anti = 0.0

    
    def update(self, output, stopflag=False, ICflag = False):
        # output must be "current value from sensor", not the actual output of the function
        # The actual output of the function is "self.steering" & "self.throttle". You should call this variable to use for car.steering & car.throttle
        # error = self.setpoint - output
        error = - output

        cur_t = datetime.now()  # needs to import `datetime`
        if self.pre_t is None:
            dt = 0.0
        else:
            dt = (cur_t - self.pre_t).total_seconds()
            # print(f"dt: {dt}")

        self.pre_t = cur_t

        if dt > 0.0:
            self.integral += (error - self.Ka * self.anti) * dt
            derivative = (error - self.pre_error) / dt
        else:
            # when dt == 0, prevent dividing zero
            derivative = 0.0

        self.pre_error = error

        self.steering = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        saturated = self.steering


        self.anti = self.steering - saturated
        self.steering = saturated

        max_throttle = self.throttle_limits[1]  # Maximum throttle value
        min_throttle = self.throttle_limits[0]  # Minimum throttle value
        self.throttle = min_throttle + (max_throttle - min_throttle) * abs(self.steering)
    