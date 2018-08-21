from math import atan
import rospy
DEG_RAD = 0.017453
class YawController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel
        
        self.min_angle = -max_steer_angle # * DEG_RAD
        self.max_angle = max_steer_angle # * DEG_RAD
    
    
    def get_angle(self, radius):
        #rospy.loginfo("calculated yaw radius: %.2f", radius)
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        #rospy.loginfo("calculated get_angle: %.2f", max(self.min_angle, min(self.max_angle, angle)))
        #rospy.loginfo("Final steering angle: %.2f, compared to max steer: %.2f",
        #max(self.min_angle, min(self.max_angle, angle)), self.max_angle)
        
        return max(self.min_angle, min(self.max_angle, angle))
    
    def get_steering(self, linear_velocity, angular_velocity, current_velocity):
        angular_velocity = current_velocity * angular_velocity / linear_velocity if abs(linear_velocity) > 0. else 0.
        
        if abs(current_velocity) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_velocity);
            angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, angular_velocity))
        #rospy.loginfo("calculated angular velocity at the given speed: %.2f", angular_velocity)
        return self.get_angle(max(current_velocity, self.min_speed) / angular_velocity) if abs(angular_velocity) > 0. else 0.0;

