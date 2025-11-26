import argparse
import os
import torch

class BaseOptions():
    """
    基础配置类：定义训练和测试通用的参数。
    这里特别加入了针对 3D 超声物理特性的参数。
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # --- 基本实验配置 ---
        parser.add_argument('--name', type=str, default='experiment_augan', help='实验名称，决定了模型保存的文件夹名')
        parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ID，例如 0 或 0,1,2。使用 -1 代表 CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型保存的根目录')
        parser.add_argument('--dataroot', type=str, default='./data_assets', help='数据集的根目录')
        
        # --- 模型架构参数 ---
        parser.add_argument('--model', type=str, default='augan', help='选择模型类型 [augan | test]')
        parser.add_argument('--input_nc', type=int, default=1, help='输入通道数 (超声通常为1)')
        parser.add_argument('--output_nc', type=int, default=1, help='输出通道数 (超声通常为1)')
        parser.add_argument('--ngf', type=int, default=64, help='生成器(G)最后一层卷积的滤波器数量')
        parser.add_argument('--ndf', type=int, default=64, help='判别器(D)第一层卷积的滤波器数量')
        parser.add_argument('--netG', type=str, default='unet_3d', help='生成器架构 [unet_3d]')
        parser.add_argument('--netD', type=str, default='pixel', help='判别器架构 [pixel | n_layers]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='判别器层数 (仅当 netD 为 n_layers 时有效)')
        parser.add_argument('--norm', type=str, default='instance', help='归一化层类型 [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='网络初始化方法 [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='初始化增益系数')
        parser.add_argument('--no_dropout', action='store_true', help='如果指定，则在生成器中不使用 Dropout')
        
        # --- [关键改造] 3D 物理与数据参数 (解耦硬编码) ---
        parser.add_argument('--patch_size_d', type=int, default=256, help='[Z轴] 深度方向的切块大小 (高分辨率轴)')
        parser.add_argument('--patch_size_h', type=int, default=64, help='[X轴] 高度方向的切块大小 (低分辨率轴)')
        parser.add_argument('--patch_size_w', type=int, default=64, help='[Y轴] 宽度方向的切块大小 (低分辨率轴)')
        
        parser.add_argument('--norm_min', type=float, default=-60.0, help='物理归一化最小值 (dB)')
        parser.add_argument('--norm_max', type=float, default=0.0, help='物理归一化最大值 (dB)')
        
        parser.add_argument('--batch_size', type=int, default=1, help='输入批次大小')
        parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
        
        # --- 其他通用参数 ---
        parser.add_argument('--suffix', default='', type=str, help='自定义后缀，用于修改实验名称')
        parser.add_argument('--verbose', action='store_true', help='如果指定，打印更多调试信息')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """初始化解析器并解析参数"""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # 获取基本选项
        opt, _ = parser.parse_known_args()

        # 修改模型相关的选项 (动态加载模型类中的参数)
        # 注意：这里需要我们在 models/__init__.py 里实现 get_option_setter
        # 为了避免现在的循环引用，我们暂时保留这个逻辑，等 models 写好再测试
        
        # 这里是一个占位符，真正运行时会调用
        # model_option_setter = models.get_option_setter(opt.model)
        # parser = model_option_setter(parser, self.isTrain)
        
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """打印并保存选项到文本文件"""
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # 保存到磁盘
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """解析参数入口"""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # 由子类 (TrainOptions/TestOptions) 设置

        # 设置 GPU
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt