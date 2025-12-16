# import pyvista as pv
# import nibabel as nib
# import numpy as np
# import os

# # =================配置区=================
# # 建议写绝对路径，防止找不到文件
# # 比如： /home/liujia/AUGAN_3D_Recon/results/....
# FILE_PATH = 'results/run_04_z1024_L1_loss/latest/nifti/Sim_0001_Pts_019_HQ.nii'

# # 核心物理参数
# VOXEL_SPACING = (0.2, 0.2, 0.041) 
# # =======================================

# def render_scifi_volume():
#     # 0. 全局设置暗黑主题 (修复 TypeError 报错)
#     pv.set_plot_theme("dark")
    
#     print(f"正在读取文件: {FILE_PATH} ...")
    
#     # 检查文件是否存在
#     if not os.path.exists(FILE_PATH):
#         print(f"❌ 错误：文件不存在！路径: {os.path.abspath(FILE_PATH)}")
#         print("⚠️  启动备用模式：生成随机数据演示效果...")
#         # 生成随机数据
#         data = np.random.rand(128, 128, 512) * 100
#         data[:, :, 200:220] += 150 
#     else:
#         nii = nib.load(FILE_PATH)
#         data = nii.get_fdata()

#     # 1. 数据归一化 (0-255)
#     data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    
#     # 2. 创建 PyVista 体积网格 (使用 ImageData 修复旧版报错)
#     grid = pv.ImageData()
#     grid.dimensions = data.shape
#     grid.spacing = VOXEL_SPACING 
#     grid.origin = (0, 0, 0)
#     grid.point_data["values"] = data.flatten(order="F")
    
#     # 3. 创建渲染器 (移除了 theme 参数)
#     plotter = pv.Plotter(window_size=[1200, 800])
#     plotter.set_background("#080808") # 手动设置背景色作为双重保险
    
#     # --- 颜色与透明度魔法 ---
#     opacity = [0,   0.0, 
#                50,  0.0, 
#                100, 0.05, 
#                180, 0.4, 
#                255, 0.9]
    
#     # # 4. 加载体积渲染
#     # vol = plotter.add_volume(
#     #     grid, 
#     #     cmap="cyan_magma", 
#     #     opacity=opacity,
#     #     shade=True,
#     #     diffuse=0.7,
#     #     specular=0.5,
#     #     specular_power=15,
#     #     blending="composite"
#     # )
#     # 4. 加载体积渲染
#     # 自定义颜色列表：黑色背景 -> 青色肌肉 -> 橙色筋膜 -> 黄色高亮
#     my_cmap = ["#000000", "#003333", "#00FFFF", "#FF4500", "#FFFF00"]

#     vol = plotter.add_volume(
#         grid, 
#         cmap=my_cmap,    # <--- 传入这个列表
#         opacity=opacity,
#         shade=True,
#         diffuse=0.7,
#         specular=0.5,
#         specular_power=15,
#         blending="composite"
#     )

#     # 5. 添加“增强现实”元素 (HUD)
#     plotter.add_text("AUGAN REAL-TIME NAV", position='upper_left', font_size=14, color='#00FFFF')
#     plotter.add_text("TARGET: SMAS LAYER", position='upper_right', font_size=12, color='#FF4500')
    
#     plotter.add_bounding_box(color='grey', line_width=1)
    
#     # 模拟“超声炮焦点”
#     center = grid.center
#     # 注意：Z轴可能很长，我们把焦点稍微往下放一点，方便看到
#     focus_z = center[2] 
#     focus_point = pv.Sphere(radius=1.5, center=(center[0], center[1], focus_z))
#     plotter.add_mesh(focus_point, color="red", style="wireframe", line_width=2, render_lines_as_tubes=True)
    
#     line = pv.Line((center[0], center[1], 0), (center[0], center[1], focus_z))
#     plotter.add_mesh(line, color="red", line_width=1)

#     print("渲染完成！")
#     print("操作提示：\n - 按住【鼠标左键】旋转\n - 按住【Shift+左键】拖动\n - 滚动【滚轮】缩放")
    
#     plotter.show()

# if __name__ == "__main__":
#     render_scifi_volume()
import pyvista as pv
import nibabel as nib
import numpy as np
import os

# =================配置区=================
# 这里建议填绝对路径
FILE_PATH = 'results/run_04_z1024_L1_loss/latest/nifti/Sim_0001_Pts_019_HQ.nii'
OUTPUT_FILENAME = "augan_scifi_demo.gif"  # 输出文件名
VOXEL_SPACING = (0.2, 0.2, 0.041) 
# =======================================

def render_scifi_volume():
    # 0. 【关键】启动虚拟显示器 (专门解决 WSL 报错)
    print("正在启动后台虚拟显卡...")
    pv.start_xvfb()
    pv.set_plot_theme("dark")
    
    print(f"正在读取文件: {FILE_PATH} ...")
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ 错误：文件不存在！")
        return
    else:
        nii = nib.load(FILE_PATH)
        data = nii.get_fdata()

    # 1. 数据归一化
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    
    # 2. 创建 PyVista 网格
    grid = pv.ImageData()
    grid.dimensions = data.shape
    grid.spacing = VOXEL_SPACING 
    grid.origin = (0, 0, 0)
    grid.point_data["values"] = data.flatten(order="F")
    
    # 3. 创建渲染器 (开启 off_screen 离线模式)
    plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
    plotter.set_background("#080808") 
    
    # 4. 颜色与透明度 (Magma 色谱)
    opacity = [0, 0.0, 50, 0.0, 100, 0.05, 180, 0.4, 255, 0.9]
    
    vol = plotter.add_volume(
        grid, 
        cmap="magma", 
        opacity=opacity,
        shade=True,
        diffuse=0.7,
        specular=0.5,
        blending="composite"
    )

    # 5. 添加 HUD
    plotter.add_text("AUGAN 3D SYSTEM", position='upper_left', font_size=10, color='#00FFFF')
    plotter.add_bounding_box(color='grey', line_width=1)
    
    # 模拟焦点
    center = grid.center
    focus_point = pv.Sphere(radius=1.5, center=center)
    plotter.add_mesh(focus_point, color="red", style="wireframe", line_width=2)
    line = pv.Line((center[0], center[1], 0), center)
    plotter.add_mesh(line, color="red", line_width=1)

    # 6. 设置相机初始视角
    plotter.camera_position = 'xz'
    plotter.camera.azimuth = 45
    plotter.camera.elevation = 30

    print(f"正在渲染并保存动图到: {OUTPUT_FILENAME} (可能需要几十秒)...")
    
    # 7. 生成旋转动画
    # 让模型旋转 360 度并保存为 GIF
    path = plotter.generate_orbital_path(n_points=36, shift=grid.length)
    plotter.open_gif(OUTPUT_FILENAME)
    
    # 旋转一圈
    plotter.orbit_on_path(path, write_frames=True)
    
    plotter.close()
    print("✅ 渲染成功！请在 Windows 文件夹中打开查看。")

if __name__ == "__main__":
    render_scifi_volume()