import argparse
import os
import torch
import models  # [!!] 关键修复: 导入 models 模块以加载模型特定参数

class BaseOptions():
    """
    基础配置类 [V5.0 - 最终修复版]
    1. 导入了 models 以解决参数加载错误。
    2. 加入了物理间距参数，支持物理比例可视化。
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # --- 基础路径与名称 ---
        parser.add_argument('--name', type=str, default='experiment_augan', help='实验名称')
        parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ID')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型保存目录')
        parser.add_argument('--dataroot', type=str, default='./project_assets/Ultrasound_Simulation_Data_500_2/04_Pair_data_1024', help='数据根目录')
        
        # --- 模型结构 ---
        parser.add_argument('--model', type=str, default='augan', help='模型类型')
        parser.add_argument('--input_nc', type=int, default=1, help='输入通道数')
        parser.add_argument('--output_nc', type=int, default=1, help='输出通道数')
        parser.add_argument('--ngf', type=int, default=64, help='G 滤波器数')
        parser.add_argument('--ndf', type=int, default=64, help='D 滤波器数')
        parser.add_argument('--netG', type=str, default='unet_3d', help='G 架构')
        parser.add_argument('--netD', type=str, default='pixel', help='D 架构')
        parser.add_argument('--n_layers_D', type=int, default=3, help='D 层数')
        parser.add_argument('--norm', type=str, default='instance', help='归一化层')
        parser.add_argument('--init_type', type=str, default='normal', help='初始化方法')
        parser.add_argument('--init_gain', type=float, default=0.02, help='初始化增益')
        parser.add_argument('--no_dropout', action='store_true', help='禁用 Dropout')
        
        # --- 数据增强 ---
        parser.add_argument('--no_flip', action='store_true', help='禁用翻转')
        parser.add_argument('--no_elastic', action='store_true', help='禁用弹性形变') # [新增] 方便做消融实验
        
        # --- [关键] 3D 物理参数 ---
        parser.add_argument('--patch_size_d', type=int, default=256, help='Z轴 Patch 大小')
        parser.add_argument('--patch_size_h', type=int, default=64, help='X轴 Patch 大小')
        parser.add_argument('--patch_size_w', type=int, default=64, help='Y轴 Patch 大小')
        
        # 默认 Z1024 参数 (Z=0.0362mm)
        parser.add_argument('--spacing_z', type=float, default=0.0362, help='Z轴物理间距 (mm)')
        parser.add_argument('--spacing_x', type=float, default=0.2, help='X轴物理间距 (mm)')
        
        parser.add_argument('--norm_min', type=float, default=-60.0, help='归一化最小值 (dB)')
        parser.add_argument('--norm_max', type=float, default=0.0, help='归一化最大值 (dB)')
        
        parser.add_argument('--batch_size', type=int, default=2, help='Batch Size')
        parser.add_argument('--num_workers', type=int, default=4, help='线程数')
        
        # --- 通用 ---
        parser.add_argument('--suffix', default='', type=str, help='自定义后缀')
        parser.add_argument('--verbose', action='store_true', help='打印详细信息')
        
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        
        # [关键] 加载模型特定参数 (如 lambda_L2)
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()
        
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
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

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
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