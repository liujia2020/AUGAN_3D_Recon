"""
数据加载模块的初始化文件。
包含 create_dataset 工厂函数，对外提供统一的数据接口。
"""
import torch
from torch.utils.data import DataLoader
from data.ultrasound_dataset import UltrasoundDataset

def create_dataset(opt):
    """
    创建并返回一个 PyTorch DataLoader。
    
    参数:
        opt: 配置参数对象
    """
    # 1. 实例化数据集 (读取文件、定义处理逻辑)
    dataset = UltrasoundDataset(opt)
    
    # 2. 决定是否打乱数据
    # 训练时：必须打乱 (Shuffle=True) 以增强泛化能力
    # 测试时：不打乱 (Shuffle=False) 以便按顺序保存结果，方便与原始文件名对应
    shuffle = True if opt.isTrain else False
    
    # 3. 创建 DataLoader (多线程加载器)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=opt.num_workers, # 线程数，由 options 控制
        pin_memory=True,             # 锁页内存，加速 CPU -> GPU 传输
        drop_last=True if opt.isTrain else False # 训练时丢弃最后不足一个 batch 的数据
    )
    
    print(f"Dataset [{type(dataset).__name__}] created. "
          f"Size: {len(dataset)}, Batch Size: {opt.batch_size}, "
          f"Shuffle: {shuffle}, Workers: {opt.num_workers}")
          
    return dataloader