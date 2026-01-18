import numpy as np

class PIDController:
    """
    PID控制器实现
    支持位置式PID控制，包含积分分离和抗饱和机制
    """
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=1.0, output_limits=(0, 100), integral_limit=None):
        """
        初始化PID控制器
        
        Args:
            kp (float): 比例系数
            ki (float): 积分系数
            kd (float): 微分系数
            dt (float): 控制周期
            output_limits (tuple): 输出限制 (min, max)
            integral_limit (float, optional): 积分限幅值
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self.integral_limit = integral_limit
        
        self.reset()

    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_output = 0.0

    def set_parameters(self, kp, ki, kd):
        """在线更新PID参数"""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def update(self, setpoint, measured_value):
        """
        计算控制输出
        
        Args:
            setpoint (float): 目标值
            measured_value (float): 当前测量值
            
        Returns:
            float: 控制输出
        """
        # 计算误差
        error = setpoint - measured_value
        
        # 积分项计算
        self.integral += error * self.dt
        
        # 积分抗饱和（如果有设置）
        if self.integral_limit:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
            
        # 微分项计算
        derivative = (error - self.prev_error) / self.dt
        
        # 计算输出
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # 输出限幅
        if self.output_limits:
            output = max(self.output_limits[0], min(self.output_limits[1], output))
            
        # 保存状态
        self.prev_error = error
        self.last_output = output
        
        return output

class MultiLoopPID:
    """
    多回路PID控制器 (用于温湿度耦合控制)
    """
    def __init__(self, temp_params, hum_params, dt=1.0):
        # 温度控制器 (假设输出为加热/通风，范围 -100 到 100)
        # 正值加热，负值降温/通风
        self.temp_pid = PIDController(
            kp=temp_params[0], ki=temp_params[1], kd=temp_params[2], 
            dt=dt, output_limits=(-100, 100)
        )
        
        # 湿度控制器 (假设输出为加湿/除湿，范围 -100 到 100)
        # 正值加湿，负值除湿
        self.hum_pid = PIDController(
            kp=hum_params[0], ki=hum_params[1], kd=hum_params[2], 
            dt=dt, output_limits=(-100, 100)
        )
        
    def set_parameters(self, params_vector):
        """
        从向量设置参数
        params_vector: [Kp_T, Ki_T, Kd_T, Kp_H, Ki_H, Kd_H]
        """
        self.temp_pid.set_parameters(params_vector[0], params_vector[1], params_vector[2])
        self.hum_pid.set_parameters(params_vector[3], params_vector[4], params_vector[5])
        
    def step(self, target_temp, current_temp, target_hum, current_hum):
        u_temp = self.temp_pid.update(target_temp, current_temp)
        u_hum = self.hum_pid.update(target_hum, current_hum)
        return u_temp, u_hum
    
    def reset(self):
        self.temp_pid.reset()
        self.hum_pid.reset()
