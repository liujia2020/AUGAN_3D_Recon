% =========================================================================
% Master_Recon_Sparse_Standalone.m
% 
% 职责：
% 1. 专门用于生成 [4个, 6个, 8个] 角度的稀疏采样重建数据。
% 2. 这是一个“独立”脚本，内置了重建算法，无需修改外部函数文件。
% 3. 自动读取 RF 数据，进行重建，并保存到新的文件夹中。
% =========================================================================

clc;
clear;
close all;

% --- 1. 定义路径 (请确保与你之前的路径一致) ---
MasterOutputDir = 'D:\Verasonics_3\Data_Acq_5\Ultrasound_Simulation_Data_500_2'; 
inputDir = fullfile(MasterOutputDir, '02_RF_Data');

% 定义稀疏角度的输出目录
Ang4BaseDir = fullfile(MasterOutputDir, '03_Recon_MAT', 'Sparse_04_Angle');
Ang6BaseDir = fullfile(MasterOutputDir, '03_Recon_MAT', 'Sparse_06_Angle');
Ang8BaseDir = fullfile(MasterOutputDir, '03_Recon_MAT', 'Sparse_08_Angle');

% 自动创建文件夹
if ~exist(Ang4BaseDir, 'dir'), mkdir(Ang4BaseDir); end
if ~exist(Ang6BaseDir, 'dir'), mkdir(Ang6BaseDir); end
if ~exist(Ang8BaseDir, 'dir'), mkdir(Ang8BaseDir); end

fprintf('--- 稀疏角度重建任务启动 ---\n');
fprintf('输出目录已准备:\n 1. %s\n 2. %s\n 3. %s\n', Ang4BaseDir, Ang6BaseDir, Ang8BaseDir);

% --- 2. 查找 RF 文件 ---
fileList = dir(fullfile(inputDir, 'SimData_RF_*.mat'));
if isempty(fileList)
    error('在 %s 中未找到 SimData_RF_*.mat 文件。', inputDir);
end
% 排序
fileNames = {fileList.name};
[~, sortIdx] = sort(fileNames);
fileList = fileList(sortIdx);

% --- 3. 定义任务列表 {角度数量, 输出文件夹名} ---
sparse_tasks = {
    4, Ang4BaseDir;
    6, Ang6BaseDir;
    8, Ang8BaseDir
};

% --- 4. 循环处理每个文件 ---
for i = 1:length(fileList)
    currentFile = fileList(i).name;
    fullFilePath = fullfile(inputDir, currentFile);
    
    fprintf('\n========================================\n');
    fprintf('正在处理文件 (%d / %d): %s\n', i, length(fileList), currentFile);
    
    % 解析文件名
    [~, baseName, ~] = fileparts(currentFile); 
    tokens = regexp(baseName, 'SimData_RF_(\d+)(_Pts_\d+)?', 'tokens');
    if isempty(tokens)
        fprintf(' > 警告: 文件名格式无法解析，跳过。\n');
        continue;
    end
    num_str = tokens{1}{1}; 
    pts_str = tokens{1}{2}; 
    outputBaseName = sprintf('Recon_Combined_%s%s', num_str, pts_str);

    % 加载数据 (只加载一次，供后续多个角度任务使用)
    dataLoaded = false; 
    loadedData = [];

    % --- 遍历 4, 6, 8 角度任务 ---
    for t = 1:size(sparse_tasks, 1)
        num_angles = sparse_tasks{t, 1};
        target_dir = sparse_tasks{t, 2};
        
        mat_filename = fullfile(target_dir, [outputBaseName, '.mat']);
        
        % 检查文件是否已存在
        if exist(mat_filename, 'file')
            fprintf(' > [跳过] %d 角度结果已存在: %s\n', num_angles, outputBaseName);
            continue;
        end
        
        % 如果需要处理且数据未加载，则加载数据
        if ~dataLoaded
            fprintf(' > 正在加载 RF 数据...\n');
            try
                loadedData = load(fullFilePath);
                dataLoaded = true;
            catch ME
                warning('加载 RF 文件失败: %s', ME.message);
                break; % 跳出任务循环，处理下一个文件
            end
        end
        
        fprintf('   >> 正在重建 [%d 角度] 模式...\n', num_angles);
        
        try
            % 调用内置的局部函数 (Local Functions)
            % 这些函数已被修改以支持指定角度数量
            [volume_CR, x_axis, y_axis, z_axis] = local_Recon_CR(...
                loadedData.RcvData, loadedData.Trans, loadedData.Resource, ...
                loadedData.TX, loadedData.TW, loadedData.Receive, num_angles);
            
            [volume_RC, ~, ~, ~] = local_Recon_RC(...
                loadedData.RcvData, loadedData.Trans, loadedData.Resource, ...
                loadedData.TX, loadedData.TW, loadedData.Receive, num_angles);
                
            if ~isequal(size(volume_CR), size(volume_RC))
                warning('CR 和 RC 尺寸不一致，跳过此任务。');
            else
                % 融合
                volume_CR_permuted = permute(volume_CR, [1 3 2]);
                volume_combined = volume_RC + volume_CR_permuted;
                
                z_space_cropped = z_axis;
                x_space_cropped = x_axis;
                y_space_cropped = y_axis;

                % 保存
                save(mat_filename, 'volume_combined', 'x_space_cropped', 'y_space_cropped', 'z_space_cropped', '-v7.3');
                fprintf('      -> 已保存: %s\n', mat_filename);
            end
        catch ME_recon
            warning('重建 [%d 角度] 时出错: %s', num_angles, ME_recon.message);
        end
    end
    
    % 清理内存
    clear loadedData volume_CR volume_RC volume_combined;
end

fprintf('\n--- 所有稀疏重建任务完成！ ---\n');


% =========================================================================
%   局部函数定义区 (Local Functions)
%   这是原 function_Recon_CR/RC.m 的修改版，直接内嵌在这里，方便独立运行
% =========================================================================

function [volume_CR, x_axis, y_axis, z_axis] = local_Recon_CR(RcvData, Trans, Resource, TX, TW, Receive, num_angles_wanted)
    % [CR 模式] 列发 - 行收
    % num_angles_wanted: 整数，例如 4, 6, 8
    
    % --- 基础参数 ---
    f0 = double(Trans.frequency*1e6);    
    fs = f0*Receive(1).samplesPerWave;   
    c0 = Resource.Parameters.speedOfSound;
    lambda = c0/f0;                      
    ElementPos = Trans.ElementPos.*lambda;
    initial_time = TW.peak/Trans.frequency/1e6;
    pitch = 0.2e-3;   
    scan.startdepth = 5e-3;
    scan.enddepth = 42e-3;

    % --- 提取数据 ---
    column_waves = length(TX)/2; % 通常是 33
    channel_RF = zeros(Receive(1).endSample,Resource.Parameters.numRcvChannels,length(TX),Resource.RcvBuffer.numFrames);
    for n_frame = 1:Resource.RcvBuffer.numFrames
        for n_wave = 1:length(TX)
            if iscell(RcvData)
                channel_RF(:,:,n_wave,n_frame) = RcvData{1}(Receive(n_wave).startSample:Receive(n_wave).endSample,Trans.Connector,n_frame);
            else
                channel_RF(:,:,n_wave,n_frame) = RcvData(Receive(n_wave).startSample:Receive(n_wave).endSample,Trans.Connector,n_frame);
            end
        end
    end
    % 提取 CR 部分
    RF_data = permute(channel_RF(:,1:128,column_waves+1:end,1), [1 3 2]); 
    RF_data = hilbert(RF_data);

    % --- [关键修改] 计算稀疏角度索引 ---
    if num_angles_wanted >= column_waves
        angle_indices = 1:column_waves; % 全角度
    else
        % 在 1 到 33 之间均匀采样
        angle_indices = round(linspace(1, column_waves, num_angles_wanted));
    end
    
    % --- FFT 设置 ---
    Nx = 128; dx = 0.0002; Nz = 1024; Nt = Receive(1).endSample;
    z_axis_temp = linspace(scan.startdepth,scan.enddepth,Nz).';
    dz = z_axis_temp(2)-z_axis_temp(1);
    cc = 1540; fHigh = 30e6; fLow = 2;
    nFFTt = 8*2^nextpow2(Nt); nFFTx = 2*2^nextpow2(Nx); nFFTz = 2*2^nextpow2(Nz);
    dOmega = fs/nFFTt;
    omega = ((0:(nFFTt-1)) - floor(nFFTt/2))'*dOmega; omega = ifftshift(omega);
    dkx = 1/dx/nFFTx;
    kx = ((0:(nFFTx-1)) - floor(nFFTx/2))*dkx; kx = ifftshift(kx);
    dkz = 1/dz; 
    kz = (fLow)/(cc) + (0:(nFFTz-1))*(dkz/nFFTz); 
    omegaBandIndex = (omega <= (fHigh)) & (omega >= (fLow)); 
    omegaBand = omega(omegaBandIndex); 
    [kx_matrix1,omegaBand_matrix] = meshgrid(kx,omegaBand);
    isevanescent = abs(omegaBand_matrix)./abs(kx_matrix1)<cc;

    % --- 重建循环 ---
    fk_data = zeros(nFFTz,nFFTx,128); 
    PkzkxX = zeros(nFFTz,nFFTx,128); 

    for ixt = angle_indices % 使用计算好的稀疏索引
        RFshow = RF_data(1:Nt,ixt,1:128); 
        RFo_y = fft(RFshow,nFFTt,1);     
        RFo_y = RFo_y(omegaBandIndex,:); 
        RFo_ky = fft(RFo_y,nFFTx,2);     
        
        angle = TX(ixt+column_waves).Steer(2);
        sinA = sin(angle); cosA = cos(angle);
        dt = -initial_time*cosA+sinA*((Nx-1)*(angle<0)-(0:Nx-1))*dx/cc;
        tmp = bsxfun(@times, omegaBand, dt);
        [kx_matrix,kz_matrix] = meshgrid(kx,kz);
        if sinA==0
            K_all = (kx_matrix.^2+kz_matrix.^2)./(2.*kx_matrix.*0+2.*kz_matrix);
        else
            K_all = (-kz_matrix*cosA+sqrt(kx_matrix.^2*sinA.^2+kz_matrix.^2))/sinA.^2;
        end
        OMEGA_st = cc*K_all;
        
        for i = 1:128 
            RFo_ky1 = RFo_ky.*repmat(exp(-2*1i*pi*tmp(:,i)),[1,nFFTx]);
            RFo_ky1(isevanescent) = 0;
            Pkzkx = zeros(size(OMEGA_st));
            for j=1:nFFTx
                Pkzkx(:,j) = interp1(omegaBand,RFo_ky1(:,j),OMEGA_st(:,j));
            end
            Akzkx = kz_matrix./K_all; Akzkx(isnan(Akzkx)) = 1;
            Pkzkx = Pkzkx.*Akzkx;
            Pkzkx(isnan(Pkzkx) | (OMEGA_st < omegaBand(1)) | (OMEGA_st > omegaBand(end))) = 0;
            
            if angle < 0 
                relative_dx = ElementPos(i,1) - ElementPos(128,1); 
                z_intersect = relative_dx/tan(angle);
                z_points = linspace(scan.startdepth, scan.enddepth, nFFTz);
                valid_mask = z_points < z_intersect;  
            elseif angle > 0 
                relative_dx = ElementPos(i,1) - ElementPos(1,1);
                z_intersect = relative_dx/tan(angle);
                z_points = double(linspace(scan.startdepth, scan.enddepth, nFFTz));
                valid_mask = z_points < z_intersect;  
            else 
                valid_mask = ones(1, nFFTz);
            end

            Pkzkx_spatial = ifft(ifft(Pkzkx,[],1),[],2);  
            Pkzkx_spatial = Pkzkx_spatial.*repmat(valid_mask', 1, size(Pkzkx_spatial,2));
            Pkzkx = fft(fft(Pkzkx_spatial,[],1),[],2);
            PkzkxX(:,:,i) = Pkzkx;
        end
        fk_data = fk_data + PkzkxX;
    end

    fk_data2 = ifft(ifft(fk_data,[],1),[],2);
    start_idx = 140; x_half = 128;
    fk_data_CR = fk_data2(start_idx:start_idx+1023, 1:x_half, :);
    
    z_space_cropped = linspace(5e-3, 42e-3, size(fk_data_CR,1)); 
    x_space_cropped = linspace(-127*pitch/2, 127*pitch/2, x_half);    
    y_space_cropped = linspace(-127*pitch/2, 127*pitch/2, size(fk_data_CR,3));
    
    volume_CR = fk_data_CR;
    x_axis = x_space_cropped; y_axis = y_space_cropped; z_axis = z_space_cropped;
end

function [volume_RC, x_axis, y_axis, z_axis] = local_Recon_RC(RcvData, Trans, Resource, TX, TW, Receive, num_angles_wanted)
    % [RC 模式] 行发 - 列收
    
    f0 = double(Trans.frequency*1e6); fs = f0*Receive(1).samplesPerWave; c0 = Resource.Parameters.speedOfSound;
    lambda = c0/f0; ElementPos = Trans.ElementPos.*lambda; initial_time = TW.peak/Trans.frequency/1e6;
    pitch = 0.2e-3; scan.startdepth = 5e-3; scan.enddepth = 42e-3;

    row_waves = length(TX)/2; 
    channel_RF = zeros(Receive(1).endSample,Resource.Parameters.numRcvChannels,length(TX),Resource.RcvBuffer.numFrames);
    for n_frame = 1:Resource.RcvBuffer.numFrames
        for n_wave = 1:length(TX)
            if iscell(RcvData)
                channel_RF(:,:,n_wave,n_frame) = RcvData{1}(Receive(n_wave).startSample:Receive(n_wave).endSample,Trans.Connector,n_frame);
            else
                channel_RF(:,:,n_wave,n_frame) = RcvData(Receive(n_wave).startSample:Receive(n_wave).endSample,Trans.Connector,n_frame);
            end
        end
    end
    % 提取 RC 部分
    RF_data = permute(channel_RF(:,129:256,1:row_waves,1), [1 3 2]); 
    RF_data = hilbert(RF_data);

    % --- [关键修改] 计算稀疏角度索引 ---
    if num_angles_wanted >= row_waves
        angle_indices = 1:row_waves;
    else
        angle_indices = round(linspace(1, row_waves, num_angles_wanted));
    end

    Nx = 128; dx = 0.0002; Nz = 1024; Nt = Receive(1).endSample;
    z_axis_temp = linspace(scan.startdepth,scan.enddepth,Nz).';
    dz = z_axis_temp(2)-z_axis_temp(1);
    cc = 1540; fHigh = 30e6; fLow = 2;
    nFFTt = 8*2^nextpow2(Nt); nFFTx = 2*2^nextpow2(Nx); nFFTz = 2*2^nextpow2(Nz);
    dOmega = fs/nFFTt;
    omega = ((0:(nFFTt-1)) - floor(nFFTt/2))'*dOmega; omega = ifftshift(omega);
    dkx = 1/dx/nFFTx;
    kx = ((0:(nFFTx-1)) - floor(nFFTx/2))*dkx; kx = ifftshift(kx);
    dkz = 1/dz; 
    kz = (fLow)/(cc) + (0:(nFFTz-1))*(dkz/nFFTz); 
    omegaBandIndex = (omega <= (fHigh)) & (omega >= (fLow)); 
    omegaBand = omega(omegaBandIndex); 
    [kx_matrix1,omegaBand_matrix] = meshgrid(kx,omegaBand);
    isevanescent = abs(omegaBand_matrix)./abs(kx_matrix1)<cc;

    fk_data = zeros(nFFTz,nFFTx,128); PkzkxX = zeros(nFFTz,nFFTx,128);

    for ixt = angle_indices 
        RFshow = RF_data(1:Nt,ixt,1:128);
        RFo_y = fft(RFshow,nFFTt,1); RFo_y = RFo_y(omegaBandIndex,:); RFo_ky = fft(RFo_y,nFFTx,2);
        
        angle = TX(ixt).Steer(1);
        sinA = sin(angle); cosA = cos(angle);
        dt = -initial_time*cosA+sinA*((Nx-1)*(angle<0)-(0:Nx-1))*dx/cc;
        tmp = bsxfun(@times, omegaBand, dt); 
        [kx_matrix,kz_matrix] = meshgrid(kx,kz);
        if sinA==0
            K_all = (kx_matrix.^2+kz_matrix.^2)./(2.*kx_matrix.*0+2.*kz_matrix);
        else
            K_all = (-kz_matrix*cosA+sqrt(kx_matrix.^2*sinA.^2+kz_matrix.^2))/sinA.^2;
        end
        OMEGA_st = cc*K_all;
        
        for i = 1:128 
            RFo_ky1 = RFo_ky.*repmat(exp(-2*1i*pi*tmp(:,i)),[1,nFFTx]);
            RFo_ky1(isevanescent) = 0;
            Pkzkx = zeros(size(OMEGA_st));
            for j=1:nFFTx
                Pkzkx(:,j) = interp1(omegaBand,RFo_ky1(:,j),OMEGA_st(:,j));
            end
            Akzkx = kz_matrix./K_all; Akzkx(isnan(Akzkx)) = 1;
            Pkzkx = Pkzkx.*Akzkx;
            Pkzkx(isnan(Pkzkx) | (OMEGA_st < omegaBand(1)) | (OMEGA_st > omegaBand(end))) = 0;

            if angle < 0 
                relative_dx = ElementPos(i,1) - ElementPos(128,1); 
                z_intersect = relative_dx/tan(angle);
                z_points = linspace(scan.startdepth, scan.enddepth, nFFTz);
                valid_mask = z_points < z_intersect;  
            elseif angle > 0 
                relative_dx = ElementPos(i,1) - ElementPos(1,1);
                z_intersect = relative_dx/tan(angle);
                z_points = double(linspace(scan.startdepth, scan.enddepth, nFFTz));
                valid_mask = z_points < z_intersect;  
            else 
                valid_mask = ones(1, nFFTz);
            end

            Pkzkx_spatial = ifft(ifft(Pkzkx,[],1),[],2);  
            Pkzkx_spatial = Pkzkx_spatial.*repmat(valid_mask', 1, size(Pkzkx_spatial,2));
            Pkzkx = fft(fft(Pkzkx_spatial,[],1),[],2);
            PkzkxX(:,:,i) = Pkzkx;
        end
        fk_data = fk_data + PkzkxX;
    end

    fk_data2 = ifft(ifft(fk_data,[],1),[],2);
    start_idx = 140; x_half = 128; 
    fk_data_RC = fk_data2(start_idx:start_idx+1023, 1:x_half, :);

    z_space_cropped = linspace(5e-3, 42e-3, size(fk_data_RC,1)); 
    x_space_cropped = linspace(-127*pitch/2, 127*pitch/2, x_half);    
    y_space_cropped = linspace(-127*pitch/2, 127*pitch/2, size(fk_data_RC,3));

    volume_RC = fk_data_RC;
    x_axis = x_space_cropped; y_axis = y_space_cropped; z_axis = z_space_cropped;
end