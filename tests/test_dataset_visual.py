import sys
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，防止 WSL 报错
import matplotlib.pyplot as plt

# --- 路径 Hack: 让脚本能找到 data 模块 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ultrasound_dataset import UltrasoundDataset

def run_test_v3():
    print("🚀 启动数据层深度核查 V3 (物理各向异性修正版)...")
    
    # --- 1. 物理参数定义 (根据你的描述) ---
    # Z轴 (深度): 1024点对应 42mm -> ~0.04mm? 
    # 你之前说是 0.0326，我们以你给的数值为准
    SPACING_Z = 0.036168  # mm 
    SPACING_X = 0.2     # mm
    SPACING_Y = 0.2     # mm
    
    # 计算 Matplotlib 显示用的纵横比 (Aspect Ratio)
    # 我们希望 Z轴 1个像素的高度，看起来只有 X轴 1个像素宽度的 ~0.16倍
    # 这样才能还原 8.35mm : 12.8mm 的物理比例
    VISUAL_ASPECT = SPACING_Z / SPACING_X  # ≈ 0.163
    
    print(f"📏 物理参数设定:")
    print(f"   Z Spacing: {SPACING_Z} mm")
    print(f"   X Spacing: {SPACING_X} mm")
    print(f"   显示纵横比 (Aspect): {VISUAL_ASPECT:.4f}")

    # --- 2. 模拟参数 ---
    class MockOpt:
        # [!!] 你的真实数据路径
        dataroot = '/home/liujia/AUGAN_Simplified/project_assets/Ultrasound_Simulation_Data_500_2/04_Pair_data_1024'
        phase = 'train'
        # Patch Size (像素单位)
        patch_size_d = 256
        patch_size_h = 64
        patch_size_w = 64
        # 归一化参数
        norm_min = -60.0
        norm_max = 0.0
        isTrain = True
        no_flip = False 
        batch_size = 1
        
    opt = MockOpt()
    
    # --- 3. 初始化数据集 ---
    try:
        dataset = UltrasoundDataset(opt)
        print(f"✅ 数据集加载成功，共 {len(dataset)} 组。")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 结果保存目录
    save_dir = './tests/output_check_v3'
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 4. 核心测试循环 ---
    # 随机取 1 个样本进行详细解剖
    idx = np.random.randint(0, len(dataset))
    print(f"\n--- 正在处理样本 Index {idx} ---")
    
    sample = dataset[idx]
    # Tensor 形状: (C, D, H, W) -> (1, 256, 64, 64)
    lq_tensor = sample['LQ']
    hq_tensor = sample['HQ']
    
    # ==========================================
    # [修正 A] 保存带有物理信息的 NIfTI
    # ==========================================
    # 1. 维度还原: (C, D, H, W) -> (D, H, W)
    lq_numpy = lq_tensor.squeeze(0).numpy()
    hq_numpy = hq_tensor.squeeze(0).numpy()
    
    # 2. 转回 NIfTI 标准顺序 (X, Y, Z)
    # 当前: (Z=256, X=64, Y=64)
    # 目标: (X=64,  Y=64, Z=256)
    # 变换: permute(1, 2, 0)
    lq_nii_data = lq_numpy.transpose(1, 2, 0)
    hq_nii_data = hq_numpy.transpose(1, 2, 0)
    
    # 3. 构建仿射矩阵 (Affine Matrix)
    # 对角线元素代表 spacing: [dx, dy, dz, 1]
    # 注意：因为数据是 (X, Y, Z) 顺序，所以 spacing 也是 (X, Y, Z)
    affine = np.diag([SPACING_X, SPACING_Y, SPACING_Z, 1.0])
    
    nii_lq = nib.Nifti1Image(lq_nii_data, affine)
    nii_hq = nib.Nifti1Image(hq_nii_data, affine)
    
    path_lq_nii = os.path.join(save_dir, f'check_idx_{idx}_LQ_phys.nii.gz')
    path_hq_nii = os.path.join(save_dir, f'check_idx_{idx}_HQ_phys.nii.gz')
    
    nib.save(nii_lq, path_lq_nii)
    nib.save(nii_hq, path_hq_nii)
    print(f"📦 已保存物理校正的 NIfTI (请用 ITK-SNAP 验证):")
    print(f"   -> {path_lq_nii}")

    # ==========================================
    # [修正 B] 物理比例还原绘图
    # ==========================================
    # 我们切一个侧面图 (Depth-Lateral Plane)
    # 取 Y轴 (W) 的中间
    w_center = lq_tensor.shape[3] // 2
    
    # 切片形状: (Depth, Height) = (256, 64)
    lq_slice = lq_tensor[0, :, :, w_center].numpy() 
    hq_slice = hq_tensor[0, :, :, w_center].numpy()
    diff = hq_slice - lq_slice

    fig, axes = plt.subplots(1, 3, figsize=(12, 5)) # 画布调矮一点
    
    titles = ['LQ (Physical Aspect)', 'HQ (Physical Aspect)', 'Diff (HQ-LQ)']
    images = [lq_slice, hq_slice, diff]
    cmaps = ['gray', 'gray', 'coolwarm']
    
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        # [核心修正] aspect=0.163
        # 让 256 的高度被“压扁”，看起来像 42
        im = ax.imshow(img, cmap=cmap, vmin=-1, vmax=1, aspect=VISUAL_ASPECT)
        ax.set_title(title)
        ax.set_xlabel("Lateral (X) [0.2mm]")
        ax.set_ylabel("Depth (Z) [0.0326mm]")
    
    plt.tight_layout()
    path_png = os.path.join(save_dir, f'check_idx_{idx}_phys_view.png')
    plt.savefig(path_png)
    plt.close()
    print(f"🖼️  物理比例还原图已保存: {path_png}")
    print(f"   (现在的图像应该是一个扁的长方形，符合 8.3mm x 12.8mm 的物理尺寸)")

if __name__ == '__main__':
    run_test_v3()