% 1. (新增) 定义新的 '01_Recon_MAT' 输出文件夹。
% 2. 自动查找 '02_RF_Data' 中的所有 .mat 文件。
% 3. 对每个 .mat 文件，执行【两次】重建 ('full' 和 'zero')。
% 4. (已修正) 使用【Permute 和 加法】进行融合。
% 5. (核心) 仅保存【高分辨率复数 .mat 文件】。
% 6. (已移除) 不再生成 NIfTI 或 PNG。
%
clc;
clear;
close all;
% --- 1. (!! 新增 !!) 定义新的、有序的输出文件夹 ---
% (这是你要求的全新主文件夹)
% MasterOutputDir = 'Ultrasound_Recon_Pipeline_FK'; 
% inputDir = 'Ultrasound_Simulation_Data\02_RF_Data'; % (数据源还是老地方)

MasterOutputDir = 'D:\Verasonics_3\Data_Acq_5\Ultrasound_Simulation_Data_500_2'; % 定义主文件夹
inputDir = fullfile(MasterOutputDir, '02_RF_Data'); % 定义子文件夹


% (新增) 定义两个 .mat 输出目录
FullAngleBaseDir = fullfile(MasterOutputDir, '03_Recon_MAT', 'Full_33_Angle');
ZeroAngleBaseDir = fullfile(MasterOutputDir, '03_Recon_MAT', 'Zero_0_Degree');

% (新增) 自动创建 .mat 文件夹s
if ~exist(FullAngleBaseDir, 'dir'), mkdir(FullAngleBaseDir); end
if ~exist(ZeroAngleBaseDir, 'dir'), mkdir(ZeroAngleBaseDir); end
fprintf('--- MAT 文件将保存到: %s ---\n', fullfile(MasterOutputDir, '01_Recon_MAT'));

% --- 2. 查找并正确排序所有 RF 文件 ---
fileList = dir(fullfile(inputDir, 'SimData_RF_*.mat'));
if isempty(fileList)
    error('在 %s 中未找到 SimData_RF_*.mat 文件。', inputDir);
end
fileNames = {fileList.name};
[~, sortIdx] = sort(fileNames);
fileList = fileList(sortIdx);
fprintf('--- 自动化重建开始，共找到 %d 个文件 ---\n', length(fileList));

% --- 3. 循环处理每个文件 ---
for i = 1:length(fileList)
    currentFile = fileList(i).name;
    fullFilePath = fullfile(inputDir, currentFile);
    fprintf('\n========================================\n');
    fprintf('正在处理文件: %s (%d / %d)\n', currentFile, i, length(fileList));
    
    % --- 4. 提取文件名 ---
    [~, baseName, ~] = fileparts(currentFile); % baseName 例: 'SimData_RF_0001_Pts_342'
    
    % [!!] 核心修改：使用更精确的正则表达式 [!!]
    % 从 'SimData_RF_0001_Pts_342' 中提取 '0001' 和 '_Pts_342'
    tokens = regexp(baseName, 'SimData_RF_(\d+)(_Pts_\d+)?', 'tokens');
    
    if isempty(tokens)
        fprintf(' > 警告: 无法从 %s 中解析编号。跳过此文件。\n', currentFile);
        continue;
    end
    
    num_str = tokens{1}{1}; % '0001'
    pts_str = tokens{1}{2}; % '_Pts_342' (如果存在)
    
    outputBaseName = sprintf('Recon_Combined_%s%s', num_str, pts_str);
    % 示例: 'Recon_Combined_0001_Pts_342'

    % --- 5. 加载 RF 数据 (只加载一次) ---
    fprintf(' > 正在加载 RF 文件...\n');
    try
        loadedData = load(fullFilePath);
        disp(' > RF 文件加载成功。');
    catch ME
        warning(ME.identifier, '加载文件 %s 失败: %s. 跳过此文件。', currentFile, ME.message);
        continue;
    end
    
    % % ==========================================================
    % % --- 流程 1: "FULL ANGLE" 重建 ---
    % % ==========================================================
    fprintf('\n--- 开始 [Full Angle] 重建 ---\n');
    
    mat_filename_full = fullfile(FullAngleBaseDir, [outputBaseName, '.mat']);
    if exist(mat_filename_full, 'file')
        fprintf(' > 已跳过 (Full Angle MAT 文件已存在): %s\n', outputBaseName);
    else
        try
            % 6-F. 调用重建函数 ('full')
            fprintf(' > 正在重建 (Full Angle)...\n');
            [volume_CR, x_axis, y_axis, z_axis] = function_Recon_CR(...
                loadedData.RcvData, loadedData.Trans, loadedData.Resource, ...
                loadedData.TX, loadedData.TW, loadedData.Receive, 'full');
            
            [volume_RC, ~, ~, ~] = function_Recon_RC(...
                loadedData.RcvData, loadedData.Trans, loadedData.Resource, ...
                loadedData.TX, loadedData.TW, loadedData.Receive, 'full');
                
            if ~isequal(size(volume_CR), size(volume_RC))
                warning('Full Angle CR 和 RC 容积大小不一致。跳过此重建。');
            else
                % 7-F. 融合 (加法)
                fprintf(' > 正在复合 (Full Angle)...\n');
                volume_CR_permuted = permute(volume_CR, [1 3 2]);
                volume_combined = volume_RC + volume_CR_permuted;
                z_space_cropped = z_axis;
                x_space_cropped = x_axis;
                y_space_cropped = y_axis;

                % 8-F. 保存 .mat 文件
                fprintf(' > 正在保存 [Full Angle] .mat 文件...\n');
                save(mat_filename_full, ...
                    'volume_combined', ...
                    'x_space_cropped', 'y_space_cropped', 'z_space_cropped', ...
                    '-v7.3');
                fprintf(' > [Full Angle] .mat 文件已保存: %s\n', mat_filename_full);
            end
        catch ME_recon_full
            warning(ME_recon_full.identifier, '重建 [Full Angle] (%s) 时出错: %s. 跳过此重建。', currentFile, ME_recon_full.message);
            disp('错误详情:');
            disp(ME_recon_full.getReport); 
        end
    end
    
    % % ==========================================================
    % % --- 流程 2: "ZERO ANGLE" 重建 ---
    % % ==========================================================
    fprintf('\n--- 开始 [Zero Angle] 重建 ---\n');
    
    mat_filename_zero = fullfile(ZeroAngleBaseDir, [outputBaseName, '.mat']);
    if exist(mat_filename_zero, 'file')
        fprintf(' > 已跳过 (Zero Angle MAT 文件已存在): %s\n', outputBaseName);
    else
        try
            % 6-Z. 调用重建函数 ('zero')
            fprintf(' > 正在重建 (Zero Angle)...\n');
            [volume_CR, x_axis, y_axis, z_axis] = function_Recon_CR(...
                loadedData.RcvData, loadedData.Trans, loadedData.Resource, ...
                loadedData.TX, loadedData.TW, loadedData.Receive, 'zero');
            
            [volume_RC, ~, ~, ~] = function_Recon_RC(...
                loadedData.RcvData, loadedData.Trans, loadedData.Resource, ...
                loadedData.TX, loadedData.TW, loadedData.Receive, 'zero');
                
            if ~isequal(size(volume_CR), size(volume_RC))
                warning('Zero Angle CR 和 RC 容积大小不一致。跳过此重建。');
            else
                % 7-Z. 融合 (加法)
                fprintf(' > 正在复合 (Zero Angle)...\n');
                volume_CR_permuted = permute(volume_CR, [1 3 2]);
                volume_combined = volume_RC + volume_CR_permuted;
                z_space_cropped = z_axis;
                x_space_cropped = x_axis;
                y_space_cropped = y_axis;

                % 8-Z. 保存 .mat 文件
                fprintf(' > 正在保存 [Zero Angle] .mat 文件...\n');
                save(mat_filename_zero, ...
                    'volume_combined', ...
                    'x_space_cropped', 'y_space_cropped', 'z_space_cropped', ...
                    '-v7.3');
                fprintf(' > [Zero Angle] .mat 文件已保存: %s\n', mat_filename_zero);
            end
        catch ME_recon_zero
            warning(ME_recon_zero.identifier, '重建 [Zero Angle] (%s) 时出错: %s. 跳过此重建。', currentFile, ME_recon_zero.message);
            disp('错误详情:');
            disp(ME_recon_zero.getReport); 
        end
    end

    % --- 10. 清理内存 ---
    fprintf('\n文件 %s 处理完毕。\n', currentFile);
    clear loadedData volume_CR volume_RC volume_combined volume_CR_permuted x_axis y_axis z_axis x_space_cropped y_space_cropped z_space_cropped;
    
end

fprintf('\n========================================\n');
fprintf('--- 自动化 MAT 重建全部完成！ ---\n');