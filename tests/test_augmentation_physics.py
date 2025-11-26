import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # 必须放在 pyplot 之前
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- 路径 Hack ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.transforms import ElasticDeformation

def create_grid_phantom(shape, spacing=10):
    """
    创建一个 3D 网格体数据。
    背景为 0 (黑)，网格线为 1 (白)。
    spacing: 网格线的间隔 (像素)
    """
    D, H, W = shape
    phantom = np.zeros(shape, dtype=np.float32)
    
    # 画 Z 轴线 (水平线)
    phantom[::spacing, :, :] = 1.0
    # 画 X 轴线 (垂直线)
    phantom[:, ::spacing, :] = 1.0
    # 画 Y 轴线
    # phantom[:, :, ::spacing] = 1.0 # 为了看清侧面，暂时只画平面网格
    
    return phantom

def run_physics_test():
    print("🚀 启动物理增强验证 (Physics Augmentation Test)...")
    
    # 1. 物理参数 (基于你的数据)
    SPACING_Z = 0.0362
    SPACING_X = 0.2
    # 比例因子: Z轴需要比 X轴平滑多少倍？
    ANISO_RATIO = SPACING_X / SPACING_Z  # ≈ 5.52
    
    VISUAL_ASPECT = SPACING_Z / SPACING_X # ≈ 0.18
    
    print(f"📏 物理参数: Z_res={SPACING_Z}, X_res={SPACING_X}")
    print(f"🌊 平滑度倍率 (Z vs X): {ANISO_RATIO:.2f}x")
    
    # 2. 创建虚拟网格 (Phantom)
    # 大小模拟一个 Patch: 256 x 64 x 64
    shape = (256, 64, 64)
    grid = create_grid_phantom(shape, spacing=8) # 每8个像素画一条线
    
    # 3. 设置增强器
    base_sigma = 50.0 # X轴基准
    
    # [方案 A] 各向同性 (旧版/错误版) - 用于对比
    sigma_iso = base_sigma
    deformer_bad = ElasticDeformation(
        np.random.RandomState(42), 
        sigma=sigma_iso, 
        alpha=2000, 
        execution_probability=1.0 # 强制执行
    )
    
    # [方案 B] 各向异性 (新版/正确版)
    # Z轴 sigma 放大 5.5 倍
    sigma_aniso = (base_sigma * ANISO_RATIO, base_sigma, base_sigma)
    deformer_good = ElasticDeformation(
        np.random.RandomState(42), # 使用相同种子以便对比
        sigma=sigma_aniso, 
        alpha=2000, 
        execution_probability=1.0
    )
    
    print(f"🛠️  生成变形中...")
    print(f"   -> 错误 Sigma (Iso): {sigma_iso}")
    print(f"   -> 正确 Sigma (Aniso): {sigma_aniso}")
    
    grid_bad = deformer_bad(grid.copy())
    grid_good = deformer_good(grid.copy())
    
    # 4. 可视化对比
    save_dir = './tests/output_physics'
    os.makedirs(save_dir, exist_ok=True)
    
    # 取中间切片 (Depth-Lateral)
    w_mid = shape[2] // 2
    slice_orig = grid[:, :, w_mid]
    slice_bad = grid_bad[:, :, w_mid]
    slice_good = grid_good[:, :, w_mid]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    
    # 原图
    axes[0].imshow(slice_orig, cmap='gray', aspect=VISUAL_ASPECT)
    axes[0].set_title("Original Grid\n(Physical Aspect)")
    axes[0].set_ylabel("Depth (Z)")
    
    # 错误的增强
    axes[1].imshow(slice_bad, cmap='gray', aspect=VISUAL_ASPECT)
    axes[1].set_title(f"Isotropic Deform (WRONG)\nSigma={base_sigma}")
    axes[1].set_xlabel("High freq jitter in Z-axis!\n(Unrealistic tissue tear)")
    
    # 正确的增强
    axes[2].imshow(slice_good, cmap='gray', aspect=VISUAL_ASPECT)
    axes[2].set_title(f"Anisotropic Deform (CORRECT)\nSigma_Z={sigma_aniso[0]:.1f}")
    axes[2].set_xlabel("Smooth Z-axis bending\n(Realistic tissue compression)")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'physics_validation.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"✅ 验证图已生成: {save_path}")
    print("   请打开图片，对比中间和右边的图。")
    print("   [中间图]: 网格线是否在纵向(Z)上剧烈抖动？(这是错的)")
    print("   [右边图]: 纵向弯曲是否变得平滑自然？(这是对的)")

if __name__ == '__main__':
    run_physics_test()