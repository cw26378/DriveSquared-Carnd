import rospy

from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, wheel_radius, steer_ratio, max_lat_accel, max_steer_angle,
                 decel_limit, vehicle_mass):

        minimum_speed = 0.1
        self.yaw_controller = YawController(wheel_base , steer_ratio , minimum_speed ,
                                            max_lat_accel , max_steer_angle)
        
        Kp = 0.3
        Ki = 0.1
        Kd = 0.0
        mn = 0.0 #clamping value for throttle
        mx = 0.25 #clamping value for throttle

        self.throttle_controller = PID(Kp, Ki, Kd , mn , mx)

        tau = 0.5 #cutoff frequency for lpf
        ts = 0.02 #filter sample time

        self.vel_lpf = LowPassFilter(tau , ts)

        self.last_time = rospy.get_time()

        self.wheel_radius = wheel_radius
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass


    def control(self, current_vel , dbw_enabled , linear_vel , angular_vel):
        if not dbw_enabled :
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)
        steering = self.yaw_controller.get_steering(linear_vel , angular_vel , current_vel)

        vel_error = linear_vel - current_vel

        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error , sample_time)
        brake = 0

        if linear_vel == 0 and current_vel < 0.1:
            throttle = 0 
            brake = 400
        elif throttle < .1 and vel_error < 0 :
            throttle = 0
            decel = max(vel_error , self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
