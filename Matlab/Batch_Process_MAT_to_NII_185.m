% % ==========================================================
% % Batch_Process_MAT_to_NII.m (V7 - MATLAB 后处理器)
% % ==========================================================
%
% 职责：
% 1. 自动查找 V6 脚本生成的 "Full_33_Angle" 和 "Zero_0_Degree" .mat 文件。
% 2. 加载 .mat 文件中的【复数】数据 (volume_combined) 和坐标轴。
% 3. (核心) 使用 imresize3 将复数数据的 Z 轴重采样到 185。
% 4. 将重采样后的数据转换为 dB 标度。
% 5. 自动创建 "Ultrasound_NIfTI_Dataset_Z185" 文件夹。
% 6. 将结果保存为配对的 'train_hq/Sim_XXXX.nii' 和 'train_lq/Sim_XXXX.nii'。
%
clc;
clear;
close all;

% --- 1. 全局配置 (请根据需要修改) ---

% (!! 修改这里 !!) 指向你的 MATLAB (V6) 输出文件夹
MATLAB_INPUT_DIR = 'D:\Verasonics_3\Data_Acq_5\Ultrasound_Simulation_Data_500_2';

% (!! 修改这里 !!) 你希望保存 NIfTI 的新位置
NIFTI_OUTPUT_DIR = 'D:\Verasonics_3\Data_Acq_5\Ultrasound_Simulation_Data_500_2/04_Pair_data_185';

% (!! 核心参数 !!)
TARGET_Z_DIM = 185;  % 你请求的目标 Z 轴维度
DB_LEVEL = -60;      % 动态范围 (dB)

% --- 2. 定义源路径和目标路径 ---
fprintf('--- MATLAB NIfTI 处理器启动 ---\n');

% 源路径 (来自 V6 脚本 )
hq_source_dir = fullfile(MATLAB_INPUT_DIR, '03_Recon_MAT', 'Full_33_Angle');
lq_source_dir = fullfile(MATLAB_INPUT_DIR, '03_Recon_MAT', 'Zero_0_Degree');

% 目标路径 (科学命名)
hq_target_dir = fullfile(NIFTI_OUTPUT_DIR, 'train_hq');
lq_target_dir = fullfile(NIFTI_OUTPUT_DIR, 'train_lq');

% 自动创建目标文件夹
if ~exist(hq_target_dir, 'dir'), mkdir(hq_target_dir); end
if ~exist(lq_target_dir, 'dir'), mkdir(lq_target_dir); end

% --- 3. 运行 HQ 和 LQ 批处理 ---
try
    process_batch(hq_source_dir, hq_target_dir, 'HQ (Full Angle)', TARGET_Z_DIM, DB_LEVEL);
    process_batch(lq_source_dir, lq_target_dir, 'LQ (Zero Angle)', TARGET_Z_DIM, DB_LEVEL);
catch ME
    fprintf('\n--- 发生严重错误: %s ---\n', ME.message);
    disp(ME.getReport);
end

fprintf('\n--- MATLAB NIfTI 处理器全部完成! ---\n');


% % ==========================================================
% % 局部函数 1: 批处理 (V7.1 - 修正体素间距计算和保存)
% % ==========================================================
function process_batch(source_dir, target_dir, data_type, target_z_dim, db_level)
    
    fprintf('\n--- 正在扫描 %s 文件夹: %s ---\n', data_type, source_dir);
    
    mat_files = dir(fullfile(source_dir, 'Recon_Combined_*.mat'));
    
    if isempty(mat_files)
        warning('在 %s 中未找到 .mat 文件。', source_dir);
        return;
    end

    for k = 1:length(mat_files)
        filename = mat_files(k).name;
        mat_path = fullfile(source_dir, filename);
        
        % [!!] 核心修改 1: 解析文件名并添加 hq/lq 标签
        % (此部分已根据你之前的需求修正)
        match = regexp(filename, 'Recon_Combined_(\d+)(_Pts_\d+)?\.mat', 'tokens');
        
        if isempty(match)
            fprintf('  [跳过] 文件名格式不匹配: %s\n', filename);
            continue;
        end
        
        file_id = match{1}{1}; % '0001'
        pts_str = match{1}{2}; % '_Pts_342' or ''

        if contains(data_type, 'HQ')
            tag = 'hq';
        elseif contains(data_type, 'LQ')
            tag = 'lq';
        else
            tag = 'unk';
        end
        
        output_filename = sprintf('Sim_%s_%s%s.nii', tag, file_id, pts_str);
        output_path = fullfile(target_dir, output_filename);
        
        if exist(output_path, 'file')
            fprintf('  [跳过] 输出文件已存在: %s\n', output_filename);
            continue;
        end
        
        fprintf('  处理: %s -> %s\n', filename, output_filename);
        
        try
            % --- A. 加载 .mat 文件 ---
            loadedData = load(mat_path);
            
            if ~isfield(loadedData, 'volume_combined')
                warning('MAT file %s is missing "volume_combined".', filename);
                continue;
            end
            
            complex_volume = loadedData.volume_combined; % (Z,X,Y) [cite: 138]
            x_axis = loadedData.x_space_cropped; % [cite: 139]
            y_axis = loadedData.y_space_cropped; % [cite: 140]
            z_axis = loadedData.z_space_cropped; % [cite: 141]
            
            original_dims = size(complex_volume); % [cite: 144]
            
            % --- B. 计算新维度 ---
            new_dims = [target_z_dim, original_dims(2), original_dims(3)]; % [cite: 147]
            
            % --- C. 重采样复数数据 ---
            real_part_resampled = imresize3(real(complex_volume), new_dims, 'linear'); % 
            imag_part_resampled = imresize3(imag(complex_volume), new_dims, 'linear'); % [cite: 152]
            resampled_complex_volume = complex(real_part_resampled, imag_part_resampled); % [cite: 154]
            
            % --- D. 转换为 dB 标度 ---
            final_nii_data = process_nii_image_local(resampled_complex_volume, db_level); % [cite: 157]
            
            % --- E. 准备 NIfTI 标头 (科学规范) ---
            
            % (为 NIfTI 标准) 将数据从 (Z,X,Y) 转换为 (X,Y,Z)
            final_nii_data_permuted = permute(final_nii_data, [2, 3, 1]); % [cite: 158]
            
            % [!!] 核心修改 2: 使用你参考脚本中的可靠计算公式
            x_res_mm = (x_axis(end) - x_axis(1)) / (length(x_axis) - 1) * 1000;
            y_res_mm = (y_axis(end) - y_axis(1)) / (length(y_axis) - 1) * 1000;
            % Z 轴间距使用可靠公式，并应用重采样缩放因子 
            z_res_original_mm = (z_axis(end) - z_axis(1)) / (length(z_axis) - 1) * 1000;
            z_res_mm = z_res_original_mm * (original_dims(1) / target_z_dim);
            
            voxel_size = [x_res_mm, y_res_mm, z_res_mm]; % (X, Y, Z 顺序)
            
            % --- F. 保存 NIfTI 文件 ---
            
            % [!!] 核心修改 3: 使用你参考脚本中的两步保存法
            
            % 1. 先使用基本参数保存
            niftiwrite(final_nii_data_permuted, output_path, 'Compressed', false);
            
            % 2. 读取刚保存的文件信息
            info = niftiinfo(output_path);
            
            % 3. 修改标头 (同时设置 PixelDimensions 和 VoxelSize)
            info.PixelDimensions = voxel_size;
            info.VoxelSize = voxel_size;
            info.Datatype = 'single'; % [cite: 172]
            info.SpaceUnits = 'Millimeter'; % [cite: 173]
            
            % 4. 重新保存
            output_path_updated = [output_path(1:end-4), '_updated.nii'];
            niftiwrite(final_nii_data_permuted, output_path_updated, info, 'Compressed', false);
            
            % 5. 替换原始文件
            movefile(output_path_updated, output_path, 'f');
            
            fprintf('   [成功] 已保存到: %s (尺寸: %s, 体素: %.4fx%.4fx%.4f mm)\n', ...
                output_path, mat2str(size(final_nii_data_permuted)), ...
                voxel_size(1), voxel_size(2), voxel_size(3));

        catch ME_Process
            warning('处理 %s 时出错: %s', filename, ME_Process.message);
            disp(ME_Process.getReport);
        end
    end
end
% % ==========================================================
% % 局部函数 2: NIfTI 处理器 (来自 V4 脚本 [cite: 551-561])
% % ==========================================================
function volume_data_nii = process_nii_image_local(volume_complex, db_level)
    % 职责: 将复数数据转换为 dB 标度的 NIfTI 数据
    volume_data_nii = abs(volume_complex);
    max_val = max(volume_data_nii(:));
    if max_val == 0
        max_val = 1; % 避免 log10(0)
    end
    volume_data_nii = 20*log10(volume_data_nii./max_val);
    volume_data_nii(volume_data_nii < db_level) = db_level; % 应用动态范围
    volume_data_nii = single(volume_data_nii);
end