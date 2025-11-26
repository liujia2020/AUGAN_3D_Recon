import sys
import os
import torch
import logging

# --- 路径 Hack ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.networks.generator import UnetGenerator3D

def run_test():
    print("🚀 启动 Generator 架构压力测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")

    # --- 测试案例配置 ---
    test_cases = [
        # (名称, 输入尺寸, 下采样次数)
        ("Standard Patch", (1, 1, 256, 64, 64), 6),  # 你的训练配置
        ("Full Volume",    (1, 1, 1024, 128, 128), 6), # 你的推理/测试配置
    ]

    for name, input_shape, num_downs in test_cases:
        print(f"\n🧪 测试场景: [{name}]")
        print(f"   输入形状: {input_shape}")
        print(f"   U-Net深度: {num_downs} 层 (递归构建)")
        
        # 1. 实例化模型
        try:
            model = UnetGenerator3D(
                input_nc=1, 
                output_nc=1, 
                num_downs=num_downs, 
                ngf=64, 
                norm_layer=torch.nn.InstanceNorm3d
            ).to(device)
            
            # 打印参数量，看看模型大小是否合理
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   ✅ 模型构建成功! 参数量: {num_params / 1e6:.2f} M")
            
        except Exception as e:
            print(f"   ❌ 模型构建失败: {e}")
            continue

        # 2. 前向传播 (Forward Pass)
        try:
            dummy_input = torch.randn(*input_shape).to(device)
            output = model(dummy_input)
            
            # 3. 维度检查
            if output.shape == input_shape:
                print(f"   ✅ 前向传播成功! 输出形状匹配: {output.shape}")
            else:
                print(f"   ❌ 维度错配! 输入 {input_shape} -> 输出 {output.shape}")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ⚠️ 显存不足 (OOM)，这是硬件限制，非代码逻辑错误。")
            else:
                print(f"   ❌ 前向传播崩溃: {e}")

if __name__ == '__main__':
    run_test()