"""
此模块实现了数据集的抽象基类 (ABC) 'BaseDataset'。
所有的具体数据集（如 UltrasoundDataset）都必须继承此类。
"""
import torch.utils.data as data
from abc import ABC, abstractmethod
import os

class BaseDataset(data.Dataset, ABC):
    """
    数据集的抽象基类。
    
    要创建一个新的数据集，你需要实现以下方法：
    -- <__init__>:                      初始化类，必须首先调用 BaseDataset.__init__(self, opt)
    -- <__len__>:                       返回数据集的大小
    -- <__getitem__>:                   获取一个数据样本
    -- <modify_commandline_options>:    (可选) 添加特定于该数据集的参数
    """

    def __init__(self, opt):
        """
        初始化基类。
        
        参数:
            opt (Option class) -- 存储所有实验标志的类
        """
        self.opt = opt
        self.root = opt.dataroot  # 从配置中获取数据根目录

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        添加特定于数据集的选项，或重写现有选项的默认值。
        
        参数:
            parser          -- 原始参数解析器
            is_train (bool) -- 是训练阶段还是测试阶段
        
        返回:
            修改后的 parser
        """
        return parser

    @abstractmethod
    def __len__(self):
        """返回数据集中的图像总数。"""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """
        返回一个数据点及其元数据信息。
        
        参数:
            index -- 数据索引
            
        返回:
            通常返回一个包含张量（Tensors）和路径信息的字典。
            例如: {'LQ': lq_tensor, 'HQ': hq_tensor, 'lq_path': path}
        """
        pass