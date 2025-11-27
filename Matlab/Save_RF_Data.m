% Save_RF_Data.m (V3 - 强同步版本)
clc;
disp('--- RF 数据保存脚本 (V3 - 强同步) ---');

% --- 1. 定义文件夹 ---
MasterOutputDir = 'Ultrasound_Simulation_Data_500_2'; % (必须与 SetUp 脚本一致)
outputDir = fullfile(MasterOutputDir, '02_RF_Data'); % 定义子文件夹
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

% --- 2. 检查 RcvData 是否存在 ---
if ~exist('RcvData', 'var')
    warning('警告：工作区中未找到 [RcvData] 变量。');
    disp('请确保 VSX 仿真已成功运行。保存已中止。');
    return; % 停止脚本
end
disp('RcvData 已在工作区找到。');

% --- 3. (!! 核心 !!) 读取通信文件 ---
comm_file = fullfile(MasterOutputDir, '__latest_config.txt');
if ~exist(comm_file, 'file')
    error('未找到通信文件: %s. SetUp 脚本可能未成功运行。', comm_file);
end

try
    fid = fopen(comm_file, 'r');
    config_filename = fgetl(fid);
    fclose(fid);
catch ME_read
    error('读取通信文件 %s 失败: %s', comm_file, ME_read.message);
end

fprintf('从通信文件中读取到 Config 文件: %s\n', config_filename);

% --- 4. (!! 核心 !!) 解析 Config 文件名以生成 RF 文件名 ---
% 示例: 从 'Config_Scene_0007_Pts_350.mat' 提取 '0007' 和 '350'
tokens = regexp(config_filename, 'Config_Scene_(\d+)_Pts_(\d+).mat', 'tokens');

if isempty(tokens)
    warning('无法解析 Config 文件名: %s. 检查正则表达式。', config_filename);
    return;
end

% 提取编号和点数
num_str = tokens{1}{1}; % '0007'
pts_str = tokens{1}{2}; % '350'
next_num = str2double(num_str);
numPoints = str2double(pts_str);

% (!! 核心 !!) 强制使用相同的编号和点数
outputFilename = sprintf('SimData_RF_%s_Pts_%s.mat', num_str, pts_str);
fullFilePath = fullfile(outputDir, outputFilename);

fprintf('将强制同步保存为: %s\n', fullFilePath);

% --- 5. 定义需要保存的变量列表 ---
varsToSave = { ...
    'RcvData', 'Receive', 'Trans', 'PData', 'Media', ...
    'Resource', 'TX', 'TW', 'TGC', 'P', 'na' ...
};

% --- 6. 保存 ---
try
    save(fullFilePath, varsToSave{:}, '-v7.3');
    fprintf('\n--- RF 数据已成功同步保存！ ---\n');
catch ME
    warning(ME.identifier, '保存文件时出错: %s', ME.message);
    disp('请检查变量列表是否正确，以及磁盘空间是否足够。');
end
return;