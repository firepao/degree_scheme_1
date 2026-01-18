import numpy as np

class VirtualGreenhouse:
    """
    虚拟温室环境模型
    用于ISSA优化过程中的预测模拟
    """
    def __init__(self, dt=1.0):
        self.dt = dt
        self.temp = 25.0
        self.rh = 60.0
        self.time_step = 0
        
        # 环境动态参数 (简化的一阶系统)
        # T_next = T + alpha*(T_out - T) + beta*u + noise
        self.alpha_t = 0.05  # 热传导率
        self.beta_t = 0.02   # 加热效率
        
        self.alpha_h = 0.03  # 湿度交换率
        self.beta_h = 0.015  # 加湿效率
        
    def reset(self, init_temp=25.0, init_rh=60.0, start_step=0):
        self.temp = init_temp
        self.rh = init_rh
        self.time_step = start_step
        return np.array([self.temp, self.rh])
    
    def get_external_weather(self, t):
        """
        模拟外部天气变化
        """
        # 简单的正弦变化模拟昼夜温差
        # 假设 24小时周期 (如果 dt=1分钟, 周期=1440)
        day_period = 1440 
        
        # 温度: 15-30度波动
        ext_temp = 22.5 + 7.5 * np.sin(2 * np.pi * t / day_period - np.pi/2)
        
        # 湿度: 50-90%波动 (温度高时湿度通常较低)
        ext_rh = 70 + 20 * np.cos(2 * np.pi * t / day_period - np.pi/2)
        
        return ext_temp, ext_rh
        
    def step(self, u_temp, u_rh):
        """
        执行一步模拟
        Args:
            u_temp (float): 温度控制信号 (-100 到 100)
            u_rh (float): 湿度控制信号 (-100 到 100)
        """
        ext_temp, ext_rh = self.get_external_weather(self.time_step)
        
        # 动态更新方程
        # 温度变化
        # 热损耗/获得 + 这里的 u_temp 包含了加热(+)和制冷(-)
        delta_temp = self.alpha_t * (ext_temp - self.temp) + self.beta_t * u_temp
        
        # 湿度变化
        # 湿气交换 + 加湿(+)/除湿(-)
        # 注意：实际上温度升高会降低相对湿度，这里为了简化暂时忽略强耦合
        delta_rh = self.alpha_h * (ext_rh - self.rh) + self.beta_h * u_rh
        
        # 添加随机噪声
        delta_temp += np.random.normal(0, 0.05)
        delta_rh += np.random.normal(0, 0.1)
        
        self.temp += delta_temp
        self.rh += delta_rh
        
        # 物理约束
        self.rh = np.clip(self.rh, 0, 100)
        self.temp = np.clip(self.temp, -10, 60)
        
        self.time_step += 1
        
        return np.array([self.temp, self.rh])

    def predict_future(self, current_state, control_sequence):
        """
        预测未来H步的状态 (用于滚动优化)
        这是ISSA算法需要的核心功能
        
        Args:
            current_state: [current_temp, current_rh]
            control_sequence: shape (H, 2) 未来H步的控制序列
            
        Returns:
            predicted_states: shape (H, 2)
        """
        temp, rh = current_state
        preds = []
        
        temp_step = self.time_step
        
        for i in range(len(control_sequence)):
            u_t, u_r = control_sequence[i]
            
            # 使用相同的物理模型进行预测 (此时不加噪声，或者是确定性预测)
            ext_temp, ext_rh = self.get_external_weather(temp_step)
            
            d_t = self.alpha_t * (ext_temp - temp) + self.beta_t * u_t
            d_r = self.alpha_h * (ext_rh - rh) + self.beta_h * u_r
            
            temp += d_t
            rh += d_r
            
            rh = np.clip(rh, 0, 100)
            temp = np.clip(temp, -10, 60)
            
            preds.append([temp, rh])
            temp_step += 1
            
        return np.array(preds)
