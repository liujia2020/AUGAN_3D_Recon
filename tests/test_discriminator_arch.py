import sys
import os
import torch
import logging

# --- 路径 Hack ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.networks.discriminator import NLayerDiscriminator3D

def run_test():
    print("🚀 启动 Discriminator 架构压力测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")

    # --- 测试案例配置 ---
    # Discriminator 的输入通常是 (Real_Image, Condition_Image) 拼接
    # 所以 input_nc 通常是 1 + 1 = 2
    test_cases = [
        # (名称, 输入尺寸, 判别器层数)
        ("Standard Patch", (1, 2, 256, 64, 64), 3),  # 训练配置
        ("Full Volume",    (1, 2, 1024, 128, 128), 3), # 推理配置(虽然D通常只在训练用)
    ]

    for name, input_shape, n_layers in test_cases:
        print(f"\n🧪 测试场景: [{name}]")
        print(f"   输入形状: {input_shape} (Batch, 2-Channels, D, H, W)")
        print(f"   层数: {n_layers}")
        
        # 1. 实例化模型
        try:
            model = NLayerDiscriminator3D(
                input_nc=2,   # 1个LQ + 1个HQ/Fake
                ndf=64,       # 基础通道数
                n_layers=n_layers, 
                norm_layer=torch.nn.InstanceNorm3d
            ).to(device)
            
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   ✅ 模型构建成功! 参数量: {num_params / 1e6:.2f} M")
            
        except Exception as e:
            print(f"   ❌ 模型构建失败: {e}")
            continue

        # 2. 前向传播
        try:
            dummy_input = torch.randn(*input_shape).to(device)
            output = model(dummy_input)
            
            print(f"   ✅ 前向传播成功! 输出形状: {output.shape}")
            
            # 3. 结果分析 (Receptive Field Check)
            # PatchGAN 经过 3 层 stride=2 的卷积，尺寸应该缩小 2^3 = 8 倍
            # 但因为它是 Valid Padding 或者是特定的 Padding 策略，尺寸可能不是严格的 /8
            expected_d = input_shape[2] // (2 ** n_layers)
            print(f"      -> 输入深度: {input_shape[2]}")
            print(f"      -> 输出深度: {output.shape[2]}")
            print(f"      -> 缩放比例: 1 : {input_shape[2] / output.shape[2]:.1f}")
            
            if output.shape[2] > 1:
                print("      -> [结论] 这是 PatchGAN (输出矩阵)，符合预期。")
            else:
                print("      -> [警告] 输出过于扁平，变成了 Vanilla GAN？")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️ 显存不足 (OOM)")
            else:
                print(f"   ❌ 前向传播崩溃: {e}")

if __name__ == '__main__':
    run_test()