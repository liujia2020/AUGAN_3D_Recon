"""
AUGAN 3D 训练主入口脚本 (V8.1 - 修复梯度报错版)
修复了 save_nii 时因未 detach 导致的 RuntimeError。
"""
import time
import os
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import nibabel as nib 

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# ==============================================================================
# [辅助函数]
# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_pixel_stats(tensor):
    return {
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.mean().item()
    }

def print_training_summary(opt, dataset, model):
    """打印详细的训练配置摘要"""
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    visual_aspect = opt.spacing_z / opt.spacing_x
    
    print("\n" + "="*80)
    print(f"{'🚀 AUGAN TRAINING CONFIGURATION':^80}")
    print("="*80)
    print(f"  - Device:        {device}")
    print(f"  - Data Root:     {opt.dataroot}")
    print(f"  - Dataset Size:  {len(dataset)} volumes")
    print(f"  - Batch Size:    {opt.batch_size}")
    print(f"  - Physics:       Z_spacing={opt.spacing_z}mm, X_spacing={opt.spacing_x}mm")
    print(f"  - Visual Aspect: {visual_aspect:.4f} (Image will be vertically compressed)")
    print(f"  - Model:         G={opt.netG}, D={opt.netD}")
    print(f"  - LR Config:     G={opt.lr}, D={opt.lr * opt.lr_d_ratio}")
    print(f"  - L2 Weight:     {opt.lambda_L2}")
    print("="*80 + "\n")

def print_epoch_report(epoch, total_epochs, epoch_time, losses_avg, lr_G, lr_D):
    """打印 Epoch 结案报告"""
    print('-' * 80)
    print(f'END OF EPOCH {epoch} / {total_epochs} \t Time Taken: {epoch_time:.0f} sec')
    print(f'  Learning Rates: \t G_lr = {lr_G:.7f} | D_lr = {lr_D:.7f}')
    
    loss_G_total = losses_avg.get('G_GAN', 0) + losses_avg.get('G_L2', 0)
    loss_D_total = (losses_avg.get('D_Real', 0) + losses_avg.get('D_Fake', 0)) * 0.5
    
    print('  Average Losses:')
    print(f'    Generator (G): \t Total ≈ {loss_G_total:.4f}')
    print(f'      ├─ G_Adversarial: \t {losses_avg.get("G_GAN", 0):.4f}')
    print(f'      └─ G_Pixelwise (L2): \t {losses_avg.get("G_L2", 0):.4f}')
    print(f'    Discriminator (D): \t Total ≈ {loss_D_total:.4f}')
    print(f'      ├─ D_Real_Loss: \t {losses_avg.get("D_Real", 0):.4f}')
    print(f'      └─ D_Fake_Loss: \t {losses_avg.get("D_Fake", 0):.4f}')
    print('-' * 80 + '\n')

def save_epoch_visuals(model, epoch, save_dir, writer, opt, save_nii=False):
    """
    保存可视化结果：
    1. PNG 图片 (物理比例矫正，横向矩形)
    2. NIfTI 文件 (带物理头信息)
    """
    visual_aspect = opt.spacing_z / opt.spacing_x
    
    # 1. 提取数据并画图 (使用 no_grad 上下文，自动处理梯度)
    with torch.no_grad():
        # 取 Batch 第一个样本的 Y 轴中间切片
        w_idx = model.real_lq.shape[4] // 2
        
        img_lq = model.real_lq[0, 0, :, :, w_idx].cpu().numpy()
        img_fake = model.fake_hq[0, 0, :, :, w_idx].cpu().numpy()
        img_real = model.real_hq[0, 0, :, :, w_idx].cpu().numpy()
        
        st_lq = get_pixel_stats(model.real_lq)
        st_fake = get_pixel_stats(model.fake_hq)
        st_real = get_pixel_stats(model.real_hq)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    t_lq = f"Input (LQ)\nRange:[{st_lq['min']:.2f}, {st_lq['max']:.2f}]"
    t_fake = f"Generated (Fake)\nRange:[{st_fake['min']:.2f}, {st_fake['max']:.2f}]"
    t_real = f"Ground Truth (HQ)\nRange:[{st_real['min']:.2f}, {st_real['max']:.2f}]"
    
    titles = [t_lq, t_fake, t_real]
    images = [img_lq, img_fake, img_real]
    
    for ax, img, title in zip(axes, images, titles):
        # 物理比例矫正
        im = ax.imshow(img, cmap='gray', vmin=-1, vmax=1, aspect=visual_aspect)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lateral (X)")
        ax.set_ylabel("Depth (Z)")
        ax.axis('on')
        
    plt.tight_layout()
    img_filename = f"epoch_{epoch:03d}.png"
    img_path = os.path.join(save_dir, img_filename)
    plt.savefig(img_path)
    plt.close(fig)
    
    writer.add_figure('Visual/Epoch_Compare', fig, global_step=epoch)
    print(f"  🖼️  Epoch {epoch} Visual Saved: {img_filename}")

    # 2. 保存 NIfTI (独立步骤)
    if save_nii:
        # [!!] 关键修复: 必须先 .detach() 再 .cpu().numpy()
        # 还原顺序: (1, 1, D, H, W) -> squeeze -> (D, H, W) -> permute -> (H, W, D)即(X, Y, Z)
        vol_fake = model.fake_hq[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
        
        # 写入物理间距
        affine = np.diag([opt.spacing_x, opt.spacing_x, opt.spacing_z, 1.0])
        nii_fake = nib.Nifti1Image(vol_fake, affine)
        
        nii_filename = f"epoch_{epoch:03d}_fake.nii.gz"
        nii_path = os.path.join(save_dir, nii_filename)
        nib.save(nii_fake, nii_path)
        print(f"  📦 NIfTI Saved: {nii_filename}")

# ==============================================================================
# [主程序]
# ==============================================================================

if __name__ == '__main__':
    # 1. 解析参数
    opt_driver = TrainOptions() 
    opt = opt_driver.parse()    
    set_seed(42)
    
    # 2. 准备目录
    log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
    img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web_images')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)

    # 3. 加载数据与模型
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    print("----------------------------------------------------------------")
    opt_driver.print_options(opt) 
    print_training_summary(opt, dataset, model)
    
    # 4. 训练循环
    total_iters = 0                
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    
    # 强制初始采样 (Step 0)
    print("📸 Saving initial sample (Step 0 check)...")
    init_batch = next(iter(dataset))
    model.set_input(init_batch)
    model.forward() 
    # 这里的 save_nii=True 会触发刚才修复的代码
    save_epoch_visuals(model, 0, img_save_dir, writer, opt, save_nii=True)
    
    for epoch in range(opt.epoch_count, total_epochs + 1):
        epoch_start_time = time.time()
        epoch_losses = {'G_GAN': 0.0, 'G_L2': 0.0, 'D_Real': 0.0, 'D_Fake': 0.0}
        epoch_iter_count = 0
        
        print(f"\nStart Epoch {epoch} / {total_epochs}")
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch}", unit="batch")

        for i, data in enumerate(progress_bar):
            total_iters += opt.batch_size
            epoch_iter_count += 1
            
            model.set_input(data)         
            model.optimize_parameters()   
            
            current_losses = model.get_current_losses()
            for k in epoch_losses.keys():
                epoch_losses[k] += current_losses.get(k, 0.0)

            if total_iters % opt.print_freq == 0:    
                progress_bar.set_postfix(G_L2=f"{current_losses['G_L2']:.3f}")
                for k, v in current_losses.items():
                    writer.add_scalar(f'Loss_Step/{k}', v, total_iters)

        # --- Epoch End ---
        avg_losses = {k: v / epoch_iter_count for k, v in epoch_losses.items()}
        for k, v in avg_losses.items():
            writer.add_scalar(f'Loss_Epoch/{k}', v, epoch)
            
        lr_G = model.optimizers[0].param_groups[0]['lr']
        lr_D = model.optimizers[1].param_groups[0]['lr']
        print_epoch_report(epoch, total_epochs, time.time() - epoch_start_time, avg_losses, lr_G, lr_D)
        
        # 每个 Epoch 必保存图和 NIfTI
        save_epoch_visuals(model, epoch, img_save_dir, writer, opt, save_nii=True)
        
        if epoch % opt.save_epoch_freq == 0:
            print(f'💾 Saving checkpoints at epoch {epoch}')
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate() 
        
    writer.close()
    print("🎉 Training Finished!")