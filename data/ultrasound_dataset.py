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
    [性能优化版] UltrasoundDataset
    核心改进：实现 Lazy Loading (懒加载)，仅读取需要的 Patch，而非整个 Volume。
    解决内存溢出导致的 IO 瓶颈。
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.dir_lq = os.path.join(opt.dataroot, opt.phase + '_lq')
        self.dir_hq = os.path.join(opt.dataroot, opt.phase + '_hq')
        
        self.lq_paths = sorted(self.make_dataset(self.dir_lq))
        self.hq_paths = sorted(self.make_dataset(self.dir_hq))
        
        self.size = len(self.lq_paths)
        
        # 目标 Patch 大小 (对应 Tensor 的 D, H, W)
        self.patch_d = opt.patch_size_d  # 256 (Z)
        self.patch_h = opt.patch_size_h  # 64 (X)
        self.patch_w = opt.patch_size_w  # 64 (Y)
        
        # 预计算各向异性 Sigma
        base_sigma = 50.0
        sigma_z = base_sigma * (SPACING_X / SPACING_Z)
        self.anisotropic_sigma = (sigma_z, base_sigma, base_sigma)
        
        logging.info(f"[{opt.phase.upper()}] Dataset initialized. Size: {self.size}")
        logging.info(f"Optimization: Lazy Loading enabled (Reading patches only).")

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
            # 1. 打开文件 (只读取 Header，不读取数据)
            lq_obj = nib.load(lq_path)
            hq_obj = nib.load(hq_path)
            
            # 获取原始形状 (X, Y, Z) = (128, 128, 1024)
            # 注意：nibabel 的 shape 属性是从 header 读的，非常快
            src_shape = lq_obj.shape 
            src_x, src_y, src_z = src_shape

            # 目标 Patch 在原始文件中的尺寸
            # 我们最终需要 (D, H, W) -> (256, 64, 64)
            # 对应原始文件的 (Z, X, Y)
            req_z = self.patch_d # 256
            req_x = self.patch_h # 64
            req_y = self.patch_w # 64

            # --- 训练模式：在硬盘上直接切片 (On-disk Slicing) ---
            if self.opt.isTrain:
                # 2. 计算随机裁剪坐标
                # 确保不越界
                if src_z < req_z or src_x < req_x or src_y < req_y:
                     return self.__getitem__(random.randint(0, self.size - 1))

                start_x = random.randint(0, src_x - req_x)
                start_y = random.randint(0, src_y - req_y)
                start_z = random.randint(0, src_z - req_z)

                # 3. [关键优化] 只读取 Patch 数据
                # 使用 dataobj 进行切片，nibabel 会智能地只读取这一块
                # 注意顺序：文件是 (X, Y, Z)
                lq_patch = lq_obj.dataobj[start_x:start_x+req_x, start_y:start_y+req_y, start_z:start_z+req_z]
                hq_patch = hq_obj.dataobj[start_x:start_x+req_x, start_y:start_y+req_y, start_z:start_z+req_z]
                
                # 4. 转为 Numpy 并修正数据类型
                lq_patch = np.array(lq_patch, dtype=np.float32)
                hq_patch = np.array(hq_patch, dtype=np.float32)

                # 5. 维度转置 (X, Y, Z) -> (Z, X, Y)
                # (64, 64, 256) -> (256, 64, 64)
                lq_patch = lq_patch.transpose(2, 0, 1)
                hq_patch = hq_patch.transpose(2, 0, 1)

                # 6. 归一化
                lq_patch = self.normalize(lq_patch)
                hq_patch = self.normalize(hq_patch)

                # 7. 数据增强 (弹性形变等)
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
                
                # 8. 截断与 Tensor
                lq_patch = np.clip(lq_patch, -1.0, 1.0)
                hq_patch = np.clip(hq_patch, -1.0, 1.0)

                lq_tensor = torch.from_numpy(lq_patch).float().unsqueeze(0)
                hq_tensor = torch.from_numpy(hq_patch).float().unsqueeze(0)

            # --- 测试模式：还是读全图 ---
            else:
                lq_vol = np.array(lq_obj.dataobj).astype(np.float32)
                hq_vol = np.array(hq_obj.dataobj).astype(np.float32)
                
                lq_vol = lq_vol.transpose(2, 0, 1)
                hq_vol = hq_vol.transpose(2, 0, 1)

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