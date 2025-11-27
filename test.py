"""
AUGAN 3D 测试主脚本 (V3.0 - 指标评估版)
功能：
1. 文件名清洗 (去除 _lq 后缀)。
2. 计算 PSNR/SSIM/MAE 并保存为 CSV。
3. 保存 NIfTI 和 PNG。
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import pandas as pd # 需要 pandas 来保存 CSV

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.metrics import calc_metrics # 导入新写的指标库

def save_visuals(real_lq, fake_hq, real_hq, img_path, save_dir, opt):
    """保存 NIfTI 和 PNG"""
    # 1. 文件名清洗 (Cleaning Filename)
    # 原始: Sim_lq_0001_Pts_019.nii -> 目标: Sim_0001_Pts_019
    short_path = os.path.basename(img_path)
    name = os.path.splitext(short_path)[0]
    
    # 核心清洗逻辑：把 '_lq' 替换为空
    clean_name = name.replace('_lq', '').replace('_hq', '')
    
    visual_aspect = opt.spacing_z / opt.spacing_x
    lq_np = real_lq.squeeze().cpu().numpy()
    fake_np = fake_hq.squeeze().cpu().numpy()
    real_np = real_hq.squeeze().cpu().numpy()
    
    # --- 保存 NIfTI ---
    if opt.save_vol_only or True:
        volumes = {'Fake': fake_np, 'LQ': lq_np, 'HQ': real_np}
        affine = np.diag([opt.spacing_x, opt.spacing_x, opt.spacing_z, 1.0])
        
        for suffix, vol_data in volumes.items():
            # (D, H, W) -> (X, Y, Z)
            vol_nii_data = vol_data.transpose(1, 2, 0)
            nii_img = nib.Nifti1Image(vol_nii_data, affine)
            
            # 文件名格式: Sim_0001_Pts_019_Fake.nii
            nii_filename = f'{clean_name}_{suffix}.nii'
            nii_save_path = os.path.join(save_dir, 'nifti', nii_filename)
            os.makedirs(os.path.dirname(nii_save_path), exist_ok=True)
            nib.save(nii_img, nii_save_path)

    # --- 保存 PNG (Y轴中间切片) ---
    w_idx = lq_np.shape[2] // 2
    img_lq = lq_np[:, :, w_idx]
    img_fake = fake_np[:, :, w_idx]
    img_real = real_np[:, :, w_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Input (LQ)', 'Generated (Fake)', 'Ground Truth (HQ)']
    images = [img_lq, img_fake, img_real]
    
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap='gray', vmin=-1, vmax=1, aspect=visual_aspect)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    # 文件名格式: Sim_0001_Pts_019_Compare.png
    png_save_path = os.path.join(save_dir, 'images', f'{clean_name}_Compare.png')
    os.makedirs(os.path.dirname(png_save_path), exist_ok=True)
    plt.savefig(png_save_path)
    plt.close(fig)
    
    return clean_name # 返回清洗后的名字用于记录

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    print("\n" + "="*80)
    print(f"🚀 STARTING TESTING: {opt.name}")
    print(f"   Physics: Z={opt.spacing_z}, X={opt.spacing_x}")
    print("="*80)
    
    save_root = os.path.join(opt.results_dir, opt.name, opt.epoch)
    os.makedirs(save_root, exist_ok=True)
    
    # 记录所有指标的列表
    metrics_list = []
    
    for i, data in enumerate(tqdm(dataset, desc="Testing")):
        model.set_input(data)
        model.test()
        
        # 1. 计算指标
        # model.fake_hq 和 model.real_hq 是 (1, 1, D, H, W)
        metrics = calc_metrics(model.fake_hq, model.real_hq)
        
        # 2. 保存图片和文件
        img_path = model.image_paths[0]
        clean_name = save_visuals(model.real_lq, model.fake_hq, model.real_hq, img_path, save_root, opt)
        
        # 3. 记录到列表
        metrics['Name'] = clean_name
        metrics_list.append(metrics)
        
    # 4. 生成报告
    df = pd.DataFrame(metrics_list)
    # 把 Name 列挪到第一列
    cols = ['Name'] + [c for c in df.columns if c != 'Name']
    df = df[cols]
    
    # 计算平均值行
    avg_row = df.mean(numeric_only=True)
    avg_row['Name'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存 CSV
    csv_path = os.path.join(save_root, 'metrics_report.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    
    print("\n" + "="*80)
    print(f"✅ TESTING FINISHED")
    print(f"   Average PSNR: {avg_row['PSNR']:.4f} dB")
    print(f"   Average SSIM: {avg_row['SSIM']:.4f}")
    print(f"   Average MAE:  {avg_row['MAE']:.4f}")
    print(f"   Report saved to: {csv_path}")
    print("="*80)