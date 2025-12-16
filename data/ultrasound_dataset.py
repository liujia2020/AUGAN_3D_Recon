import os
import torch
import numpy as np
import nibabel as nib
import random
import logging
from data.base_dataset import BaseDataset
# [修复1] 移除不存在的 get_transform，只保留 get_3d_transform
from data.transforms import  ElasticDeformation

class UltrasoundDataset(BaseDataset):
    """
    [V10.0 - 最终融合版]
    集大成者：
    1. 递归扫描 (Recursive Walk) - 解决文件夹分层
    2. 权重采样 (Weighted Sampling) - 解决数据不平衡
    3. 懒加载 (Lazy Loading) - 解决内存爆炸 (关键!)
    4. 主动采样 (Active Sampling) - 解决空 Patch 问题
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.phase_folder = opt.phase + '_lq' # e.g., 'train_lq'
        
        # [功能1] 递归扫描所有子目录
        self.lq_paths, self.hq_paths = self._walk_and_find_images(opt.dataroot, self.phase_folder)
        
        self.size = len(self.lq_paths)
        if self.size == 0:
            raise RuntimeError(f"Found 0 images in {opt.dataroot} recursively searching for {self.phase_folder}")

        logging.info(f"[{opt.phase.upper()}] Dataset initialized. Size: {self.size}")

        # Patch 参数
        self.patch_d = opt.patch_size_d 
        self.patch_h = opt.patch_size_h 
        self.patch_w = opt.patch_size_w 
        
        # 物理参数
        self.spacing_z = getattr(opt, 'spacing_z', 0.0362)
        self.spacing_x = getattr(opt, 'spacing_x', 0.2)
        
        # [参数] 信号阈值 (用于主动采样)
        self.signal_threshold = -0.95

        # 预计算各向异性增强参数
        base_sigma = 50.0
        sigma_z = base_sigma * (self.spacing_x / self.spacing_z)
        self.anisotropic_sigma = (sigma_z, base_sigma, base_sigma)

    def _walk_and_find_images(self, root_dir, target_folder_name):
        """递归遍历寻找匹配的文件夹"""
        lq_paths = []
        hq_paths = []
        IMG_EXTENSIONS = ['.nii', '.nii.gz']

        for root, dirs, files in os.walk(root_dir):
            folder_name = os.path.basename(root)
            # 只要文件夹名字对上了 (train_lq)
            if folder_name == target_folder_name:
                for fname in files:
                    if any(fname.endswith(ext) for ext in IMG_EXTENSIONS):
                        lq_p = os.path.join(root, fname)
                        # 假设命名规则严格对应: .../train_lq/... -> .../train_hq/...
                        hq_p = lq_p.replace('_lq', '_hq')
                        
                        if os.path.exists(hq_p):
                            lq_paths.append(lq_p)
                            hq_paths.append(hq_p)
        
        return sorted(lq_paths), sorted(hq_paths)

    def get_sample_weights(self):
        """[功能2] 计算采样权重 (组织 vs 点靶)"""
        # 关键词列表 (转为小写匹配)
        keywords = ['muscle', 'carotid', 'phantom', '肌肉', '颈动脉', '仿体']
        
        is_tissue_list = []
        count_tissue = 0
        count_point = 0
        
        for path in self.lq_paths:
            path_lower = path.lower()
            if any(k in path_lower for k in keywords):
                is_tissue_list.append(True)
                count_tissue += 1
            else:
                is_tissue_list.append(False)
                count_point += 1
        
        total = len(self.lq_paths)
        if count_tissue == 0: count_tissue = 1
        if count_point == 0: count_point = 1
        
        w_tissue = total / count_tissue
        w_point = total / count_point
        
        logging.info(f"⚖️ Balancing Weights: Tissue={w_tissue:.2f} (N={count_tissue}), Point={w_point:.2f} (N={count_point})")
        
        weights = [w_tissue if is_t else w_point for is_t in is_tissue_list]
        return torch.DoubleTensor(weights)

    def normalize(self, volume):
        # 全局归一化 [-1, 1]
        min_val = self.opt.norm_min
        max_val = self.opt.norm_max
        volume = np.clip(volume, min_val, max_val)
        return 2.0 * (volume - min_val) / (max_val - min_val) - 1.0

    def __getitem__(self, index):
        # [重要] 保持 Lazy Loading 逻辑
        curr_index = index % self.size
        max_retries = 20
        retries = 0

        while retries < max_retries:
            lq_path = self.lq_paths[curr_index]
            hq_path = self.hq_paths[curr_index]

            try:
                # 1. 只读 Header，不读数据
                lq_obj = nib.load(lq_path)
                hq_obj = nib.load(hq_path)
                src_x, src_y, src_z = lq_obj.shape 

                req_z, req_x, req_y = self.patch_d, self.patch_h, self.patch_w

                # --- 训练模式：Active Sampling ---
                if self.opt.isTrain:
                    if src_z < req_z or src_x < req_x or src_y < req_y:
                        curr_index = random.randint(0, self.size - 1)
                        retries += 1
                        continue

                    # 随机坐标
                    start_x = random.randint(0, src_x - req_x)
                    start_y = random.randint(0, src_y - req_y)
                    start_z = random.randint(0, src_z - req_z)

                    # [功能4] 主动采样安检
                    # 使用 dataobj 切片 (不读全图 -> 内存安全)
                    hq_slice = hq_obj.dataobj[start_x:start_x+req_x, start_y:start_y+req_y, start_z:start_z+req_z]
                    hq_patch = np.array(hq_slice, dtype=np.float32)
                    hq_patch_norm = self.normalize(hq_patch)

                    if hq_patch_norm.max() < self.signal_threshold:
                        # 是空 Patch，重试
                        if random.random() > 0.5: curr_index = random.randint(0, self.size - 1)
                        retries += 1
                        continue
                    
                    # 读取 LQ
                    lq_slice = lq_obj.dataobj[start_x:start_x+req_x, start_y:start_y+req_y, start_z:start_z+req_z]
                    lq_patch = np.array(lq_slice, dtype=np.float32)

                    # 转置: (X, Y, Z) -> (Z, X, Y)
                    lq_patch = lq_patch.transpose(2, 0, 1)
                    hq_patch = hq_patch.transpose(2, 0, 1) # 这里用未归一化的 hq_patch

                    # 归一化
                    lq_patch = self.normalize(lq_patch)
                    hq_patch = self.normalize(hq_patch) # 再次归一化

                    # 数据增强 (可选)
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

                # --- 测试模式 ---
                else:
                    lq_vol = np.array(lq_obj.dataobj).astype(np.float32).transpose(2, 0, 1)
                    hq_vol = np.array(hq_obj.dataobj).astype(np.float32).transpose(2, 0, 1)
                    lq_patch = self.normalize(lq_vol)
                    hq_patch = self.normalize(hq_vol)

                # 转 Tensor
                lq_patch = np.clip(lq_patch, -1.0, 1.0)
                hq_patch = np.clip(hq_patch, -1.0, 1.0)
                lq_tensor = torch.from_numpy(lq_patch).float().unsqueeze(0)
                hq_tensor = torch.from_numpy(hq_patch).float().unsqueeze(0)

                return {'LQ': lq_tensor, 'HQ': hq_tensor, 'lq_path': lq_path, 'hq_path': hq_path}

            except Exception as e:
                logging.warning(f"Error loading {lq_path}: {e}")
                curr_index = random.randint(0, self.size - 1)
                retries += 1
        
        return self.__getitem__(random.randint(0, self.size - 1))

    def __len__(self):
        return self.size