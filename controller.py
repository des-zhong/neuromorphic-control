import numpy as np
import torch
import casadi as ca


def rotate(vector, theta):
    x, y = vector
    x_local = np.cos(theta) * x + np.sin(theta) * y
    y_local = -np.sin(theta) * x + np.cos(theta) * y
    return np.array([x_local, y_local])

class PIDController:
    def __init__(self, c=1):
        """
        Initialize PID Controller.

        dt: Time step
        pid_params: Tuple containing (kp, ki, kd) for PID gains.
        """
        self.dt = 0.01
        self.kp_y = 5
        self.ki_y = 5
        self.kd_y = 0.05
        self.kp_x = 100
        self.ki_x = 50
        self.kd_x = 5
        self.c = c
        
        self.integral = 0  # Integral term (accumulated error)
        self.prev_error = 0  # Previous error for derivative term
    
    def pid_update(self, error, kp, ki, kd):
        """
        Update PID controller with the given error.
        
        error: The current error (sliding surface)
        
        Returns: Control signal.
        """
        # Proportional term (P)
        p_term = kp * error

        # Integral term (I)
        self.integral += error * self.dt
        i_term = ki * self.integral

        # Derivative term (D)
        d_term = kd * (error - self.prev_error) / self.dt if self.dt > 0 else 0
        
        # Update the previous error
        self.prev_error = error

        # Combine the PID terms
        control_signal = p_term + i_term + d_term
        return -control_signal
    def compute_controls(self, current_state, current_velocity, desired_state, desired_velocity):
        """
        Compute the throttle and steering commands.
        """
        # Extract states
        x, y, theta = current_state[:3]
        vx, vy = current_velocity[:2]
        x_des, y_des = desired_state[:2]
        vx_des, vy_des = desired_velocity[:2] 
        # Errors
        
        e_y = (y - y_des) + self.c * (vy - vy_des)
        e_x = (x - x_des) + self.c * (vx - vx_des)
        [e_x, e_y] = rotate([e_x, e_y], theta)
        # STA updates
        steer_control = self.pid_update(e_y, self.kp_y, self.ki_y, self.kd_y)
        throttle_control = self.pid_update(e_x, self.kp_x, self.ki_x, self.kd_x)
        return steer_control, throttle_control, e_x, e_y

class STAController:
    
    def __init__(self, c, k1_x=200, k2_x=10, kd_x=0, k1_y=0.1, k2_y=0, kd_y = 0.1):
        # scale = 1
        # STA gains
        self.k1_y = k1_y
        self.k2_y = k2_y
        self.kd_y = kd_y
        self.k1_x = k1_x
        self.k2_x = k2_x
        self.kd_x = kd_x
        self.prev_e_lat = 0
        self.prev_e_lon = 0

        # STA internal states
        self.sigma_y = 0.0
        self.sigma_x = 0.0
        self.c = c
        # Time step
        self.dt = 0.01

    def sta_update(self, sliding_surface, sigma, k1, k2):
        """
        Super-Twisting Algorithm update.
        """
        sigma_dot = k2 * np.sign(sliding_surface)
        sigma_new = sigma + sigma_dot * self.dt
        control = k1 * np.sqrt(abs(sliding_surface)) * np.sign(sliding_surface) + sigma_new
        return control, sigma_new

    def compute_controls(self, current_state, current_velocity, desired_state, desired_velocity):
        """
        Compute the throttle and steering commands.
        """
        # Extract states
        x, y, theta = current_state[:3]
        vx, vy = current_velocity[:2]
        x_des, y_des = desired_state[:2]
        vx_des, vy_des = desired_velocity[:2]
        
        # Errors
        e_x = (x_des - x) + self.c * (vx_des - vx)
        e_y = (y_des - y) + self.c * (vy_des - vy)
        [e_lon, e_lat] = rotate([e_x, e_y], theta)
        [test_lon, test_lat] = rotate([x_des - x, y_des - y], theta)

        angle_a = np.arctan2(current_velocity[1], current_velocity[0]) 
        angle_d = np.arctan2(desired_velocity[1], desired_velocity[0])

        e_theta = (angle_d - theta + np.pi) % (2 * np.pi) - np.pi

        # STA updates
        throttle_control, self.sigma_x = self.sta_update(e_lon, self.sigma_x, self.k1_x, self.k2_x)
        steer_control, self.sigma_y = self.sta_update(e_lat, self.sigma_y, self.k1_y, self.k2_y)
        
        steer_control += self.kd_y * (e_lat - self.prev_e_lat) / self.dt


        throttle_control += self.kd_x * (e_lon - self.prev_e_lon) / self.dt
        self.prev_e_lon = e_lon
        self.prev_e_lat = e_lat
        return throttle_control, steer_control, test_lon, test_lat



