import torch
import torch.nn as nn
from torch.nn import init
from .base_model import BaseModel
from .networks.generator import UnetGenerator3D
from .networks.discriminator import NLayerDiscriminator3D

class AuganModel(BaseModel):
    """
    AUGAN 模型核心类 (3D Pix2Pix 架构)。
    
    工作流:
    1. 接收 LQ (Low Quality) 和 HQ (High Quality) 3D 数据。
    2. Generator 尝试从 LQ 生成 Fake_HQ。
    3. Discriminator 尝试区分 (LQ + HQ) 和 (LQ + Fake_HQ)。
    4. 双方博弈，优化生成质量。
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加模型特有的命令行参数"""
        # 默认使用 3D U-Net 和 3D PatchGAN
        parser.set_defaults(norm='instance', netG='unet_3d', netD='pixel') 
        
        if is_train:
            # 像素级损失权重 (L2 Loss Weight)
            parser.add_argument('--lambda_L2', type=float, default=100.0, help='L2 (MSE) 像素损失的权重')
        return parser

    def __init__(self, opt):
        """初始化模型"""
        BaseModel.__init__(self, opt)
        
        # 指定需要保存/打印的损失名称
        # G_GAN: 生成器对抗损失, G_L2: 像素误差, D_Real/Fake: 判别器损失
        self.loss_names = ['G_GAN', 'G_L2', 'D_Real', 'D_Fake']
        
        # 指定可视化图片的名称 (用于 tensorboard 或 网页显示)
        self.visual_names = ['real_lq', 'fake_hq', 'real_hq']
        
        # 指定需要保存的模型 (G 和 D)
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']  # 测试时只需要生成器

        # --- 1. 定义 Generator ---
        # 动态获取 input/output channel (通常为 1)
        self.netG = UnetGenerator3D(opt.input_nc, opt.output_nc, 
                                    num_downs=6, # [物理策略] 固定为 6 层，适配 64x64 Patch
                                    ngf=opt.ngf, 
                                    norm_layer=nn.InstanceNorm3d, 
                                    use_dropout=not opt.no_dropout)
        
        # 初始化权重并移至 GPU
        self.netG = init_net(self.netG, opt.init_type, opt.init_gain, self.gpu_ids)

        # --- 2. 定义 Discriminator (仅训练时) ---
        if self.isTrain:
            # 输入通道 = LQ通道 + HQ通道 (Conditional GAN 需要看输入条件)
            netD_input_nc = opt.input_nc + opt.output_nc
            self.netD = NLayerDiscriminator3D(netD_input_nc, 
                                              ndf=opt.ndf, 
                                              n_layers=3, 
                                              norm_layer=nn.InstanceNorm3d)
            self.netD = init_net(self.netD, opt.init_type, opt.init_gain, self.gpu_ids)

        # --- 3. 定义 Loss 和 Optimizers ---
        if self.isTrain:
            # 对抗损失 (GAN Loss)
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            # 像素损失 (L2 / MSE) - 用于恢复结构和纹理
            self.criterionL2 = torch.nn.MSELoss()
            
            # 优化器
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            # [物理策略] D 的学习率可以设得比 G 低，防止 D 太强导致 G 无法学习
            # lr_d_ratio 默认为 1.0 (在 train_options.py 中定义)
            lr_D = opt.lr * opt.lr_d_ratio
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_D, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """从 DataLoader 解包数据"""
        # 数据集返回的是 {'LQ': ..., 'HQ': ..., 'lq_path': ...}
        self.real_lq = input['LQ'].to(self.device)
        self.real_hq = input['HQ'].to(self.device)
        self.image_paths = input['lq_path'] # 用于日志记录文件名

    def forward(self):
        """前向传播: G(LQ) -> Fake_HQ"""
        self.fake_hq = self.netG(self.real_lq)

    def backward_D(self):
        """反向传播 Discriminator"""
        # 1. Fake Loss: D(LQ + Fake_HQ) 应该判为 False
        # 拼接条件图 (LQ) 和 生成图 (Fake)
        # detach() 很重要！我们不想更新 G 的梯度，只想更新 D
        fake_AB = torch.cat((self.real_lq, self.fake_hq), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_Fake = self.criterionGAN(pred_fake, False)

        # 2. Real Loss: D(LQ + HQ) 应该判为 True
        real_AB = torch.cat((self.real_lq, self.real_hq), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_Real = self.criterionGAN(pred_real, True)

        # 3. Combine
        self.loss_D = (self.loss_D_Fake + self.loss_D_Real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """反向传播 Generator"""
        # 1. GAN Loss: D(LQ + Fake_HQ) 应该被判为 True (欺骗 D)
        fake_AB = torch.cat((self.real_lq, self.fake_hq), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # 2. Pixel Loss: Fake_HQ 应该无限接近 Real_HQ
        # lambda_L2 控制这一项的权重 (例如 100.0)
        self.loss_G_L2 = self.criterionL2(self.fake_hq, self.real_hq) * self.opt.lambda_L2
        
        # 3. Combine
        self.loss_G = self.loss_G_GAN + self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):
        """一步优化"""
        self.forward()                   # 计算 G(A)
        
        # 更新 D
        self.set_requires_grad(self.netD, True)  # 启用 D 的梯度
        self.optimizer_D.zero_grad()     # 清空 D 的梯度
        self.backward_D()                # 计算 D 的反向传播
        self.optimizer_D.step()          # 更新 D 的参数

        # 更新 G
        self.set_requires_grad(self.netD, False) # 冻结 D (计算 G 梯度时不需要 D 更新)
        self.optimizer_G.zero_grad()     # 清空 G 的梯度
        self.backward_G()                # 计算 G 的反向传播
        self.optimizer_G.step()          # 更新 G 的参数


# --- 辅助类与函数 ---

class GANLoss(nn.Module):
    """封装不同的 GAN 损失函数 (Vanilla, LSGAN, WGAN)"""
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def init_weights(net, init_type='normal', init_gain=0.02):
    """网络权重初始化函数"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """辅助函数: 初始化网络并推送到 GPU"""
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # 如果需要多卡并行，可以在这里加 DataParallel
        # net = torch.nn.DataParallel(net, gpu_ids) 
    init_weights(net, init_type, init_gain=init_gain)
    return net