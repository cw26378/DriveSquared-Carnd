
import rospy
import math

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_SPEED = 0.1

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband,
                 decel_limit, accel_limit,
                 wheel_radius, wheel_base, steer_ratio,
                 max_lat_accel, max_steer_angle):
        # TODO: Implement
        ## yaw controller
        self.yaw_controller = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel,                                              max_steer_angle)
        
        ## pid controller for velocity throttle control
        kp = 0.3
        ki = 0.1
        kd = 0.0
        min_throttle = 0
        max_throttle = 0.2
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)
        
        ## low pass filter
        tau = 0.5 # 1/(2Pi*tau) = cut-off frequency
        ts = 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau,ts)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.steer_ratio = steer_ratio
        
        self.last_time = rospy.get_time()
        self.last_vel = None
    
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        if not dbw_enabled:
            # reset the integral value of PID controller to zero (dealing with incorrect accumulation)
            self.throttle_controller.reset()
            # stay put when not dbw_enabled
            return 0., 0., 0.
        
        current_vel = self.vel_lpf.filt(current_vel)
        
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        #steering = angular_vel * self.steer_ratio
        
        rospy.loginfo("calculated steering angle from yaw_controller: %.2f", steering)
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        
        if linear_vel == 0 and current_vel < MIN_SPEED:
            brake = 700 # torque N*m
            throttle = 0
        
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
            
    return throttle, brake, steering

#return 1., 0., 0.

