clc;
clear;
close all;

% --- 1. 定义输出文件夹 ---
MasterOutputDir = 'D:\Verasonics_3\Data_Acq_5\Ultrasound_Simulation_Data_500_2'; % 定义主文件夹
inputDir = fullfile(MasterOutputDir, '02_RF_Data'); % 定义子文件夹

% 定义两个 .mat 输出目录
FullAngleBaseDir = fullfile(MasterOutputDir, '03_Recon_MAT', 'Full_33_Angle');
ZeroAngleBaseDir = fullfile(MasterOutputDir, '03_Recon_MAT', 'Zero_0_Degree');

% 自动创建 .mat 文件夹
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
totalFiles = length(fileList);

fprintf('--- 自动化重建准备就绪，共找到 %d 个文件 ---\n', totalFiles);

% =========================================================================
% --- [新增功能]：带超时的起始位置选择 ---
% =========================================================================
startIdx = 1; % 默认起始位置

% 提示用户
fprintf('\n------------------------------------------------------\n');
fprintf('【等待输入】请输入起始文件序号（例如 500）并回车。\n');
fprintf('           如果 5 秒内无输入，将自动从第 1 个开始。\n');
fprintf('------------------------------------------------------\n');
fprintf('起始序号 (默认 1): ');

% 创建一个定时器，5秒后自动模拟按下 "Enter" 键
% 注意：这需要 Java 支持 (MATLAB 默认都有)
t = timer;
t.StartDelay = 5; 
t.TimerFcn = @(~,~) executeEnterKey(); % 调用底部的辅助函数或直接执行
start(t);

try
    % 等待用户输入 (如果在 Timer 触发前用户回车，Timer 会被后面的 stop 停止)
    userInput = input('', 's'); 
    
    % 如果代码运行到这里，说明已经接收到了回车信号（用户按的，或者 Timer 模拟按的）
    stop(t); delete(t); % 立即停止并删除定时器
    
    if ~isempty(userInput)
        val = str2double(userInput);
        if ~isnan(val) && val >= 1 && val <= totalFiles
            startIdx = round(val);
            fprintf(' -> 用户手动指定：从第 %d 个文件开始。\n', startIdx);
        else
            fprintf(' -> 输入无效或为空，将使用默认值：从第 1 个开始。\n');
        end
    else
        fprintf(' -> (超时或空输入) 使用默认值：从第 1 个开始。\n');
    end
catch
    % 防止出错导致定时器卡死
    if exist('t', 'var'), stop(t); delete(t); end
    startIdx = 1;
end
fprintf('------------------------------------------------------\n\n');

% =========================================================================
% --- 3. 循环处理每个文件 (从 startIdx 开始) ---
% =========================================================================

for i = startIdx : totalFiles
    currentFile = fileList(i).name;
    fullFilePath = fullfile(inputDir, currentFile);
    fprintf('\n========================================\n');
    fprintf('正在处理文件: %s (%d / %d)\n', currentFile, i, totalFiles);
    
    % --- 4. 提取文件名 ---
    [~, baseName, ~] = fileparts(currentFile); 
    
    tokens = regexp(baseName, 'SimData_RF_(\d+)(_Pts_\d+)?', 'tokens');
    
    if isempty(tokens)
        fprintf(' > 警告: 无法从 %s 中解析编号。跳过此文件。\n', currentFile);
        continue;
    end
    
    num_str = tokens{1}{1}; 
    pts_str = tokens{1}{2}; 
    
    outputBaseName = sprintf('Recon_Combined_%s%s', num_str, pts_str);

    % --- 5. 加载 RF 数据 (只加载一次) ---
    % 检查是否两个输出都已经存在，如果都存在，则根本不需要加载 RF 数据 (节省时间)
    mat_filename_full = fullfile(FullAngleBaseDir, [outputBaseName, '.mat']);
    mat_filename_zero = fullfile(ZeroAngleBaseDir, [outputBaseName, '.mat']);
    
    if exist(mat_filename_full, 'file') && exist(mat_filename_zero, 'file')
        fprintf(' > [跳过] Full 和 Zero 结果均已存在，跳过加载 RF 数据。\n');
        continue; % 直接进入下一个循环
    end

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
    if exist(mat_filename_full, 'file')
        fprintf(' > 已跳过 (Full Angle MAT 文件已存在): %s\n', outputBaseName);
    else
        try
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
                % 融合
                volume_CR_permuted = permute(volume_CR, [1 3 2]);
                volume_combined = volume_RC + volume_CR_permuted;
                z_space_cropped = z_axis;
                x_space_cropped = x_axis;
                y_space_cropped = y_axis;

                save(mat_filename_full, ...
                    'volume_combined', ...
                    'x_space_cropped', 'y_space_cropped', 'z_space_cropped', ...
                    '-v7.3');
                fprintf(' > [Full Angle] .mat 文件已保存。\n');
            end
        catch ME_recon_full
            warning('重建 [Full Angle] 时出错: %s', ME_recon_full.message);
        end
    end
    
    % % ==========================================================
    % % --- 流程 2: "ZERO ANGLE" 重建 ---
    % % ==========================================================
    if exist(mat_filename_zero, 'file')
        fprintf(' > 已跳过 (Zero Angle MAT 文件已存在): %s\n', outputBaseName);
    else
        try
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
                % 融合
                volume_CR_permuted = permute(volume_CR, [1 3 2]);
                volume_combined = volume_RC + volume_CR_permuted;
                z_space_cropped = z_axis;
                x_space_cropped = x_axis;
                y_space_cropped = y_axis;

                save(mat_filename_zero, ...
                    'volume_combined', ...
                    'x_space_cropped', 'y_space_cropped', 'z_space_cropped', ...
                    '-v7.3');
                fprintf(' > [Zero Angle] .mat 文件已保存。\n');
            end
        catch ME_recon_zero
            warning('重建 [Zero Angle] 时出错: %s', ME_recon_zero.message);
        end
    end

    % --- 10. 清理内存 ---
    clear loadedData volume_CR volume_RC volume_combined volume_CR_permuted x_axis y_axis z_axis;
    
end

fprintf('\n========================================\n');
fprintf('--- 自动化 MAT 重建全部完成！ ---\n');

% -----------------------------------------------------------
% 辅助函数：模拟按下回车键 (必须放在文件末尾)
% -----------------------------------------------------------
function executeEnterKey()
    import java.awt.Robot;
    import java.awt.event.KeyEvent;
    try
        robot = Robot;
        robot.keyPress(KeyEvent.VK_ENTER);
        robot.keyRelease(KeyEvent.VK_ENTER);
    catch
        fprintf('\n(Java Robot 调用失败，请手动回车)\n');
    end
end