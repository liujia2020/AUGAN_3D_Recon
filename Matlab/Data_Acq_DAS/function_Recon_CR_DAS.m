% % ==========================================================
% % function_Recon_CR_DAS.m (列发行收 - DAS)
% % ==========================================================
function [volume_CR] = function_Recon_CR_DAS(RcvData, Trans, Resource, TX, TW, Receive, scan)
% 职责: 
% 1. 接收 RF 数据和【统一的 scan 网格】。
% 2. 执行 CR (列发行收) DAS 重建。
% 3. (已修正) 循环所有角度 (原脚本只循环1次)。
% 4. (已修正) 使用正确的 beta 角度进行变迹。
% 5. 返回重建后的容积。

disp(' [Function] 正在执行: 列发行收 (CR) DAS 重建...');

% --- 1. 提取基本参数 ---
f0 = double(Trans.frequency*1e6);
fs = f0*Receive(1).samplesPerWave;
c0 = Resource.Parameters.speedOfSound;
lambda = c0/f0;
ElementPos = Trans.ElementPos.*lambda;

% --- 2. 提取 CR 模式的 RF 数据和角度 ---
steer = zeros(length(TX),2);
for n_wave = 1:length(TX)
    steer(n_wave,:) = TX(n_wave).Steer;
end
column_waves = length(TX)/2;
beta = steer(column_waves+1:end,2); % 列发射角度

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
RF_data = channel_RF(:,1:128,column_waves+1:end,1); % CR 数据
RF_data = hilbert(RF_data);
disp(' [Function] RF 数据提取完毕。');

% --- 3. 计算 CR 接收延迟 (使用传入的 scan) ---
% (scan 结构体由 Master_Recon_Loop_DAS 定义并传入)
pitch = 0.2e-3;
RRh_probe.x = ElementPos(1:128,1);
RRh_probe.z = ElementPos(1:128,3);

Rxm = RRh_probe.x.'-scan.x(:);
Rzm = RRh_probe.z.'-scan.z(:);
receive_delay = single(sqrt(Rxm.^2+Rzm.^2)/c0); % 行接收延时

% (时间向量)
offset_distance = TW.peak*lambda;
initial_time = 0;
time_vector = initial_time+(0:(size(RF_data,1)-1))/fs;
N_elements = 128; 
array_length = pitch * (N_elements-1); 

% --- 4. CR DAS 核心循环 ---
disp(' [Function] 开始 CR DAS 重建循环...');
tic;
column_Th_row_Rh_data = zeros(scan.N_pixels,128,'single'); % 使用 'single' 节约内存

% (!!! 关键修正 1: 循环所有角度，而不是只循环1次 !!!)
for n_wave = 1:column_waves % (33 个角度)
    
    % (发射延迟)
    transmit_delay = scan.z(:)*cos(beta(n_wave))+scan.y(:)*sin(beta(n_wave));
    
    % (发射变迹)
    left_boundary = (scan.y - array_length/2)./scan.z; % (修正：你的原始脚本左右边界反了)
    right_boundary = (scan.y + array_length/2)./scan.z;
    
    % (!!! 关键修正 2: 使用 beta 角度, 而不是 alpha !!!)
    condition1 = left_boundary < tan(beta(n_wave));
    condition2 = right_boundary > tan(beta(n_wave));
    
    transmit_apodization = condition1 & condition2;
    transmit_apodization_1 = transmit_apodization(:);
    
    for n_rx = 1:128 % (128 个接收通道)
        delay = receive_delay(:,n_rx)+transmit_delay./c0;
        temp = transmit_apodization_1.*interp1(time_vector,RF_data(:,n_rx,n_wave),delay,'linear',0);
        column_Th_row_Rh_data(:,n_rx) = column_Th_row_Rh_data(:,n_rx) + single(temp);
    end
end
das_time = toc;
fprintf(' [Function] CR DAS 循环完成, 耗时: %.2f秒\n', das_time);

% --- 5. 融合通道并返回 ---
cr_data = sum(column_Th_row_Rh_data, 2);    
volume_CR = reshape(cr_data, [scan.N_z,scan.N_x,scan.N_y]);    

disp(' [Function] CR DAS 重建完成。');
end