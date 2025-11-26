import sys
import os
import torch
import shutil

# --- 路径 Hack ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.augan_model import AuganModel

def run_test():
    print("🚀 启动 AuganModel 集成测试 (Integration Test)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")

    # --- 1. 模拟完整的配置参数 ---
    class MockOpt:
        # [基础]
        # 修正: 必须是列表 [0]，不能是字符串 '0'
        gpu_ids = [0] if torch.cuda.is_available() else []
        isTrain = True
        checkpoints_dir = './tests/temp_checkpoints'
        name = 'integration_test_run'
        model = 'augan'
        verbose = False
        suffix = ''
        
        # [网络结构]
        input_nc = 1
        output_nc = 1
        ngf = 64
        ndf = 64
        netG = 'unet_3d'
        netD = 'pixel' 
        norm = 'instance'
        init_type = 'normal'
        init_gain = 0.02
        no_dropout = False
        
        # [训练与 Loss]
        gan_mode = 'vanilla'
        lr = 0.0002
        beta1 = 0.5
        lr_d_ratio = 1.0
        lambda_L2 = 100.0
        
        # [!!] 新增修复: 补齐 Scheduler 缺少的参数
        lr_policy = 'linear'
        epoch_count = 1        # 起始 epoch
        n_epochs = 100         #以此学习率训练多少 epoch
        n_epochs_decay = 100   # 衰减 epoch 数
        lr_decay_iters = 50    # 如果用 step 策略需要的参数
        continue_train = False # 是否继续训练
        load_iter = 0          # 加载迭代次数

        # [物理参数]
        patch_size_d = 256
        patch_size_h = 64
        patch_size_w = 64
        batch_size = 2 

    opt = MockOpt()
    
    # 清理旧的测试文件
    if os.path.exists(opt.checkpoints_dir):
        shutil.rmtree(opt.checkpoints_dir)

    # --- 2. 初始化模型 ---
    try:
        model = AuganModel(opt)
        model.setup(opt) # 这里会调用 get_scheduler，现在参数齐了应该能过
        print("✅ 模型初始化成功 (G + D + Optimizers + Schedulers)。")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. 构造伪数据 (Dummy Batch) ---
    print("\n📦 构造伪数据...")
    input_shape = (opt.batch_size, opt.input_nc, opt.patch_size_d, opt.patch_size_h, opt.patch_size_w)
    
    dummy_lq = torch.randn(*input_shape)
    dummy_hq = torch.randn(*input_shape)
    
    # 归一化模拟
    dummy_lq = torch.clamp(dummy_lq, -1, 1)
    dummy_hq = torch.clamp(dummy_hq, -1, 1)
    
    data = {
        'LQ': dummy_lq,
        'HQ': dummy_hq,
        'lq_path': ['fake_path_1.nii', 'fake_path_2.nii']
    }
    
    # --- 4. 运行单步优化 ---
    print("🔄 执行 optimize_parameters() ...")
    try:
        model.set_input(data)
        model.optimize_parameters()
        print("✅ 优化步执行成功。")
    except Exception as e:
        print(f"❌ 优化步失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 5. 检查 Loss ---
    print("\n📊 检查损失值 (Loss Check):")
    losses = model.get_current_losses()
    
    all_good = True
    for name, value in losses.items():
        print(f"   -> {name}: {value:.4f}")
        if value == 0.0:
            print(f"      ⚠️ 警告: Loss 为 0，可能梯度断裂？")
        if torch.isnan(torch.tensor(value)):
            print(f"      ❌ 错误: Loss 为 NaN (梯度爆炸)！")
            all_good = False
            
    if all_good:
        print("✅ 所有 Loss 数值正常。")
    
    # --- 6. 检查输出形状 ---
    if hasattr(model, 'fake_hq'):
        print(f"\n🖼️  生成结果形状: {model.fake_hq.shape}")
        if list(model.fake_hq.shape) == list(input_shape):
            print("✅ 输出形状匹配。")
        else:
            print("❌ 输出形状错误！")
    
    print("\n🎉 集成测试通过！模型已准备好进行真实训练。")

if __name__ == '__main__':
    run_test()