from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    此类仅包含训练阶段特有的参数。
    它继承自 BaseOptions，所以你在 base 里定义的参数（如 patch_size），这里都能用。
    """
    
    def initialize(self, parser):
        # 1. 先加载父类 (BaseOptions) 的通用参数
        parser = BaseOptions.initialize(self, parser)
        
        # --- 训练可视化与日志 (Visualization & Logging) ---
        parser.add_argument('--display_freq', type=int, default=100, help='每隔多少次迭代(iteration)在屏幕/日志中显示一次结果')
        parser.add_argument('--print_freq', type=int, default=20, help='每隔多少次迭代在终端打印一次 Loss')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='每隔多少次迭代保存一次 "latest" 模型 (防崩溃)')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='每隔多少个 epoch 保存一次模型检查点')
        parser.add_argument('--save_by_iter', action='store_true', help='如果开启，模型将按迭代次数保存，而不是按 epoch 保存')
        
        # --- 训练周期与策略 (Training Schedule) ---
        parser.add_argument('--continue_train', action='store_true', help='如果指定，程序会尝试加载最新的 checkpoints 继续训练')
        parser.add_argument('--epoch_count', type=int, default=1, help='起始 epoch 计数 (主要用于断点续训时调整显示)')
        parser.add_argument('--phase', type=str, default='train', help='当前阶段 [train, val, test] (数据加载器会用到)')
        
        # 核心：epoch 策略
        # 典型的 Pix2Pix 策略：前 n_epochs 保持学习率不变，后 n_epochs_decay 线性衰减至 0
        parser.add_argument('--n_epochs', type=int, default=100, help='以固定初始学习率训练的 epoch 数')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='学习率线性衰减到 0 的 epoch 数')
        
        # --- 优化器参数 (Optimizer) ---
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam 优化器的 momentum term beta1')
        parser.add_argument('--lr', type=float, default=0.0002, help='生成器 (G) 的初始学习率')
        
        # [战术优化] 判别器 (D) 往往学得比 G 快，导致 G 梯度消失。
        # 这是一个解耦的高级参数，允许你手动调低 D 的学习率。
        parser.add_argument('--lr_d_ratio', type=float, default=1.0, help='D 的学习率相对于 G 的比率 (例如 0.5 代表 D_lr = 0.5 * G_lr)')
        
        parser.add_argument('--lr_policy', type=str, default='linear', help='学习率衰减策略 [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='每多少步衰减一次学习率 (仅针对 step 策略)')
        
        # --- GAN 损失函数选择 ---
        # vanilla = 原始 GAN (sigmoid + BCE)
        # lsgan = 最小二乘 GAN (MSE)，通常训练更稳定
        # wgangp = Wasserstein GAN + 梯度惩罚，理论上最稳定但慢
        parser.add_argument('--gan_mode', type=str, default='vanilla', help='GAN 损失类型 [vanilla | lsgan | wgangp]')

        self.isTrain = True
        return parser