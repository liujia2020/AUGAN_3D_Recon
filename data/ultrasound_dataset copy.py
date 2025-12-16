import os
import torch
import numpy as np
import nibabel as nib
import random
import logging
from data.base_dataset import BaseDataset
from data.transforms import ElasticDeformation, RandomContrast

# --- 物理参数常量 ---
SPACING_Z = 0.0362
SPACING_X = 0.2
SPACING_Y = 0.2

class UltrasoundDataset(BaseDataset):
    """
    [V9.0 - 拒绝无效训练版]
    核心特性：
    1. Lazy Loading: 仅读取 Patch，内存零负担。
    2. Active Sampling: 拒绝纯黑背景 Patch，强制模型学习弱信号。
    3. Physics Aware: 各向异性增强。
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.dir_lq = os.path.join(opt.dataroot, opt.phase + '_lq')
        self.dir_hq = os.path.join(opt.dataroot, opt.phase + '_hq')
        
        self.lq_paths = sorted(self.make_dataset(self.dir_lq))
        self.hq_paths = sorted(self.make_dataset(self.dir_hq))
        
        self.size = len(self.lq_paths)
        
        # Patch 大小
        self.patch_d = opt.patch_size_d 
        self.patch_h = opt.patch_size_h 
        self.patch_w = opt.patch_size_w 
        
        # 各向异性 Sigma
        base_sigma = 50.0
        sigma_z = base_sigma * (SPACING_X / SPACING_Z)
        self.anisotropic_sigma = (sigma_z, base_sigma, base_sigma)
        
        # [关键参数] 有效信号阈值
        # 归一化后的数据范围是 [-1, 1]。
        # -0.95 意味着只要 Patch 里有一点点信号 (>-54dB)，就不算空。
        self.signal_threshold = -0.95 
        
        logging.info(f"[{opt.phase.upper()}] Dataset initialized. Size: {self.size}")
        if opt.isTrain:
            logging.info(f"Strategy: Active Sampling Enabled (Threshold > {self.signal_threshold})")

    def make_dataset(self, dir_path):
        images = []
        if not os.path.isdir(dir_path):
            logging.error(f"Directory not found: {dir_path}")
            return images
        for root, _, fnames in sorted(os.walk(dir_path)):
            for fname in fnames:
                if fname.endswith('.nii') or fname.endswith('.nii.gz'):
                    images.append(os.path.join(root, fname))
        return images

    def normalize(self, volume):
        min_val = self.opt.norm_min
        max_val = self.opt.norm_max
        volume = np.clip(volume, min_val, max_val)
        range_val = max_val - min_val
        return 2.0 * (volume - min_val) / range_val - 1.0

    def __getitem__(self, index):
        curr_index = index % self.size
        
        # 最大重试次数 (防止极少数全是黑的文件导致死循环)
        max_retries = 20 
        retries = 0

        while retries < max_retries:
            lq_path = self.lq_paths[curr_index]
            hq_path = self.hq_paths[curr_index]

            try:
                # 1. 打开文件 (Lazy Load，不读入内存)
                lq_obj = nib.load(lq_path)
                hq_obj = nib.load(hq_path)
                
                # 获取原始形状 (X, Y, Z) = (128, 128, 1024)
                src_shape = lq_obj.shape 
                src_x, src_y, src_z = src_shape 

                # 目标 Patch 尺寸 (对应 Z, X, Y)
                req_z = self.patch_d # 256
                req_x = self.patch_h # 64
                req_y = self.patch_w # 64

                # --- 训练模式：Active Sampling ---
                if self.opt.isTrain:
                    # 检查尺寸是否足够
                    if src_z < req_z or src_x < req_x or src_y < req_y:
                        curr_index = random.randint(0, self.size - 1)
                        retries += 1
                        continue

                    # 随机坐标
                    start_x = random.randint(0, src_x - req_x)
                    start_y = random.randint(0, src_y - req_y)
                    start_z = random.randint(0, src_z - req_z)

                    # [步骤 A] 先只读 HQ Patch 来做安检
                    # 使用 dataobj 直接切片 (速度极快)
                    # 注意顺序: nibabel (X, Y, Z) -> 这里的切片也是 (X, Y, Z)
                    hq_data_slice = hq_obj.dataobj[start_x:start_x+req_x, start_y:start_y+req_y, start_z:start_z+req_z]
                    hq_patch = np.array(hq_data_slice, dtype=np.float32)
                    
                    # 归一化 (为了统一数值标准)
                    hq_patch_norm = self.normalize(hq_patch)
                    
                    # [步骤 B] 核心检查：是否为空？
                    # 如果 Patch 中最大值小于阈值，说明全是背景，扔掉重来
                    if hq_patch_norm.max() < self.signal_threshold:
                        # 换个位置或换个文件重试
                        if random.random() > 0.5:
                            curr_index = random.randint(0, self.size - 1)
                        # else: 保持 curr_index 不变，重新随机 crop
                        retries += 1
                        continue
                    
                    # [步骤 C] 通关！读取对应的 LQ Patch 并继续处理
                    lq_data_slice = lq_obj.dataobj[start_x:start_x+req_x, start_y:start_y+req_y, start_z:start_z+req_z]
                    lq_patch = np.array(lq_data_slice, dtype=np.float32)
                    
                    # 维度转置 (X, Y, Z) -> (Z, X, Y) 以适配 Tensor (D, H, W)
                    lq_patch = lq_patch.transpose(2, 0, 1)
                    hq_patch = hq_patch.transpose(2, 0, 1) # 使用原始未归一化的 hq_patch 进行转置

                    # 重新归一化 (保持流程一致)
                    lq_patch = self.normalize(lq_patch)
                    hq_patch = self.normalize(hq_patch)

                    # 数据增强 (各向异性)
                    if not self.opt.no_elastic:
                        seed = np.random.randint(0, 2**32 - 1)
                        deformer_lq = ElasticDeformation(np.random.RandomState(seed), sigma=self.anisotropic_sigma)
                        deformer_hq = ElasticDeformation(np.random.RandomState(seed), sigma=self.anisotropic_sigma)
                        lq_patch = deformer_lq(lq_patch)
                        hq_patch = deformer_hq(hq_patch)

                    if not self.opt.no_flip:
                        if random.random() > 0.5:
                            lq_patch = np.flip(lq_patch, axis=2).copy()
                            hq_patch = np.flip(hq_patch, axis=2).copy()
                    
                    # 截断与 Tensor 封装
                    lq_patch = np.clip(lq_patch, -1.0, 1.0)
                    hq_patch = np.clip(hq_patch, -1.0, 1.0)

                    lq_tensor = torch.from_numpy(lq_patch).float().unsqueeze(0)
                    hq_tensor = torch.from_numpy(hq_patch).float().unsqueeze(0)
                    
                    return {'LQ': lq_tensor, 'HQ': hq_tensor, 'lq_path': lq_path, 'hq_path': hq_path}

                # --- 测试模式：全图读取 ---
                else:
                    lq_vol = np.array(lq_obj.dataobj).astype(np.float32).transpose(2, 0, 1)
                    hq_vol = np.array(hq_obj.dataobj).astype(np.float32).transpose(2, 0, 1)
                    lq_vol = self.normalize(lq_vol)
                    hq_vol = self.normalize(hq_vol)
                    lq_tensor = torch.from_numpy(lq_vol).float().unsqueeze(0)
                    hq_tensor = torch.from_numpy(hq_vol).float().unsqueeze(0)
                    return {'LQ': lq_tensor, 'HQ': hq_tensor, 'lq_path': lq_path, 'hq_path': hq_path}

            except Exception as e:
                logging.warning(f"Error loading {lq_path}: {e}. Retrying...")
                curr_index = random.randint(0, self.size - 1)
                retries += 1
        
        # 如果极其倒霉，连续 20 次都切到空图，就随便返回最后一次的结果（避免程序崩溃）
        # 这种情况在合理的数据集上极少发生
        return self.__getitem__(random.randint(0, self.size - 1))

    def __len__(self):
        return self.size