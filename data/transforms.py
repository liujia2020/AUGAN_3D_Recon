import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import logging

class ElasticDeformation:
    """
    3D 弹性形变增强 (各向异性感知版)。
    """
    def __init__(self, random_state, spline_order=3, alpha=2000, sigma=50, execution_probability=0.5, apply_3d=True):
        """
        参数:
            sigma: 高斯平滑核的大小。
                   如果是标量 (e.g., 50)，则各向同性。
                   如果是元组 (e.g., (300, 50, 50))，则分别对应 (Z, X, Y) 轴的平滑度。
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma  # Scipy 的 gaussian_filter 自动支持标量或元组
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, volume):
        if self.random_state.uniform() < self.execution_probability:
            if volume.ndim == 3:
                volume_shape = volume.shape
            else:
                volume_shape = volume[0].shape

            # 1. 生成随机位移场
            # 这里传递 self.sigma (可能是元组) 给 gaussian_filter
            # 从而实现 Z 轴更平滑 (更大的 sigma)，X/Y 轴较普通 (较小的 sigma)
            
            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros(volume_shape)

            dy, dx = [
                gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
                for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing="ij")
            indices = z + dz, y + dy, x + dx

            if volume.ndim == 3:
                return map_coordinates(volume, indices, order=self.spline_order, mode="reflect")
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode="reflect") for c in volume]
                return np.stack(channels, axis=0)
        
        return volume

# RandomContrast 类保持不变，不需要修改
class RandomContrast:
    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.5):
        self.random_state = random_state
        self.alpha = alpha 
        self.mean = mean   
        self.execution_probability = execution_probability

    def __call__(self, volume):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (volume - self.mean)
            return np.clip(result, -1, 1)
        return volume