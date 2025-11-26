import os
import torch
import numpy as np
import nibabel as nib
import random
import logging
from data.base_dataset import BaseDataset
from data.transforms import ElasticDeformation, RandomContrast

# --- [新增] 物理参数常量 ---
# 基于 Matlab 代码分析确认的数值
SPACING_Z = 0.0362  # mm (深度方向)
SPACING_X = 0.2     # mm (侧向)
SPACING_Y = 0.2     # mm (仰角)

class UltrasoundDataset(BaseDataset):
    """
    专门用于加载 3D 超声数据的数据集 (支持各向异性物理增强)。
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.dir_lq = os.path.join(opt.dataroot, opt.phase + '_lq')
        self.dir_hq = os.path.join(opt.dataroot, opt.phase + '_hq')
        
        self.lq_paths = sorted(self.make_dataset(self.dir_lq))
        self.hq_paths = sorted(self.make_dataset(self.dir_hq))
        
        if len(self.lq_paths) != len(self.hq_paths):
            logging.warning(f"数据数量不匹配! LQ: {len(self.lq_paths)}, HQ: {len(self.hq_paths)}")
            
        self.size = len(self.lq_paths)
        
        self.patch_d = opt.patch_size_d  # Z轴 (256)
        self.patch_h = opt.patch_size_h  # X轴 (64)
        self.patch_w = opt.patch_size_w  # Y轴 (64)
        
        # --- [新增] 计算各向异性 Sigma ---
        # 目标：在所有方向上保持约 10mm 的物理平滑度
        # 基准：Lateral (X) 轴 Sigma = 50 像素 -> 对应 50 * 0.2 = 10mm
        base_sigma = 50.0
        
        # Z 轴需要的 Sigma = 10mm / 0.0362mm ≈ 276 像素
        sigma_z = base_sigma * (SPACING_X / SPACING_Z)
        sigma_x = base_sigma
        sigma_y = base_sigma
        
        # 最终的各向异性 Sigma 元组 (Z, X, Y)
        self.anisotropic_sigma = (sigma_z, sigma_x, sigma_y)
        
        logging.info(f"[{opt.phase.upper()}] Dataset initialized. Found {self.size} pairs.")
        logging.info(f"Physics Spacing: Z={SPACING_Z:.4f}, X={SPACING_X:.2f}")
        logging.info(f"Anisotropic Sigma calculated: {self.anisotropic_sigma} (Z-smoothing boosted by {SPACING_X/SPACING_Z:.2f}x)")

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
        index = index % self.size
        lq_path = self.lq_paths[index]
        hq_path = self.hq_paths[index] 

        try:
            # 1. 读取 NIfTI (X, Y, Z)
            lq_obj = nib.load(lq_path)
            hq_obj = nib.load(hq_path)
            
            lq_vol = lq_obj.get_fdata().astype(np.float32)
            hq_vol = hq_obj.get_fdata().astype(np.float32)

            # 2. 维度转置 -> (Z, X, Y)
            lq_vol = lq_vol.transpose(2, 0, 1)
            hq_vol = hq_vol.transpose(2, 0, 1)

            D, H, W = lq_vol.shape

            # --- 训练模式 ---
            if self.opt.isTrain:
                if D < self.patch_d or H < self.patch_h or W < self.patch_w:
                    return self.__getitem__(random.randint(0, self.size - 1))

                # 3. 随机裁剪
                d_s = random.randint(0, D - self.patch_d)
                h_s = random.randint(0, H - self.patch_h)
                w_s = random.randint(0, W - self.patch_w)

                lq_patch = lq_vol[d_s:d_s+self.patch_d, h_s:h_s+self.patch_h, w_s:w_s+self.patch_w]
                hq_patch = hq_vol[d_s:d_s+self.patch_d, h_s:h_s+self.patch_h, w_s:w_s+self.patch_w]

                # 4. 归一化
                lq_patch = self.normalize(lq_patch)
                hq_patch = self.normalize(hq_patch)

                # 5. [关键] 各向异性物理增强
                seed = np.random.randint(0, 2**32 - 1)
                
                # 使用计算好的 anisotropic_sigma
                deformer_lq = ElasticDeformation(np.random.RandomState(seed), sigma=self.anisotropic_sigma)
                deformer_hq = ElasticDeformation(np.random.RandomState(seed), sigma=self.anisotropic_sigma)
                
                lq_patch = deformer_lq(lq_patch)
                hq_patch = deformer_hq(hq_patch)

                # 6. 翻转
                if not self.opt.no_flip:
                    if random.random() > 0.5:
                        lq_patch = np.flip(lq_patch, axis=2).copy()
                        hq_patch = np.flip(hq_patch, axis=2).copy()
                
                # 7. 强制截断 (Fix Overshoot)
                lq_patch = np.clip(lq_patch, -1.0, 1.0)
                hq_patch = np.clip(hq_patch, -1.0, 1.0)

                lq_tensor = torch.from_numpy(lq_patch).float().unsqueeze(0)
                hq_tensor = torch.from_numpy(hq_patch).float().unsqueeze(0)

            # --- 测试模式 ---
            else:
                lq_vol = self.normalize(lq_vol)
                hq_vol = self.normalize(hq_vol)
                lq_tensor = torch.from_numpy(lq_vol).float().unsqueeze(0)
                hq_tensor = torch.from_numpy(hq_vol).float().unsqueeze(0)

            return {'LQ': lq_tensor, 'HQ': hq_tensor, 'lq_path': lq_path, 'hq_path': hq_path}

        except Exception as e:
            logging.error(f"Error loading {lq_path}: {e}")
            return self.__getitem__(random.randint(0, self.size - 1))

    def __len__(self):
        return self.size