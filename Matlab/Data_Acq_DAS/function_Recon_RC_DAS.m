% % ==========================================================
% % function_Recon_RC_DAS.m (行发列收 - DAS)
% % ==========================================================
function [volume_RC] = function_Recon_RC_DAS(RcvData, Trans, Resource, TX, TW, Receive, scan)
% 职责: 
% 1. 接收 RF 数据和【统一的 scan 网格】。
% 2. 执行 RC (行发列收) DAS 重建。
% 3. 返回重建后的容积。

disp(' [Function] 正在执行: 行发列收 (RC) DAS 重建...');

% --- 1. 提取基本参数 ---
f0 = double(Trans.frequency*1e6);
fs = f0*Receive(1).samplesPerWave;
c0 = Resource.Parameters.speedOfSound;
lambda = c0/f0;
ElementPos = Trans.ElementPos.*lambda;

% --- 2. 提取 RC 模式的 RF 数据和角度 ---
steer = zeros(length(TX),2);
for n_wave = 1:length(TX)
    steer(n_wave,:) = TX(n_wave).Steer;
end
row_waves = length(TX)/2;
alpha = steer(1:row_waves,1); % 行发射角度

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
RF_data = channel_RF(:,129:256,1:row_waves,1); % RC 数据
RF_data = hilbert(RF_data);
disp(' [Function] RF 数据提取完毕。');

% --- 3. 计算 RC 接收延迟 (使用传入的 scan) ---
% (scan 结构体由 Master_Recon_Loop_DAS 定义并传入)
pitch = 0.2e-3;
CRh_probe.x = ElementPos(129:256,1);
CRh_probe.y = ElementPos(129:256,2); 
CRh_probe.z = ElementPos(129:256,3);

Cxm = CRh_probe.x.'-scan.x(:); 
Cym = CRh_probe.y.'-scan.y(:); 
Czm = CRh_probe.z.'-scan.z(:); 
receive_delay=single(sqrt(Cym.^2+Czm.^2)/c0); % 列接受延时

% (时间向量)
offset_distance=TW.peak*lambda;
initial_time = 0;
time_vector = initial_time+(0:(size(RF_data,1)-1))/fs;
N_elements = 128; 
array_length = pitch * (N_elements-1); 

% --- 4. RC DAS 核心循环 ---
disp(' [Function] 开始 RC DAS 重建循环...');
tic;
row_Th_column_Rh_data = zeros(scan.N_pixels,128,'single'); % 使用 'single' 节约内存

for n_wave = 1:row_waves % (33 个角度)
    
    % (发射延迟)
    transmit_delay = scan.z(:)*cos(alpha(n_wave))+scan.x(:)*sin(alpha(n_wave));
    
    % (发射变迹)
    left_boundary = (scan.x - array_length/2)./scan.z; % (修正：你的原始脚本左右边界反了)
    right_boundary = (scan.x + array_length/2)./scan.z;
    condition1 = left_boundary < tan(alpha(n_wave));
    condition2 = right_boundary > tan(alpha(n_wave));
    transmit_apodization = condition1 & condition2;
    transmit_apodization_1 = transmit_apodization(:);
    
    for n_rx = 1:128 % (128 个接收通道)
        delay = receive_delay(:,n_rx)+transmit_delay./c0;
        temp = transmit_apodization_1.*interp1(time_vector,RF_data(:,n_rx,n_wave),delay,'linear',0);
        row_Th_column_Rh_data(:,n_rx) = row_Th_column_Rh_data(:,n_rx) + single(temp);
    end
end
das_time = toc;
fprintf(' [Function] RC DAS 循环完成, 耗时: %.2f秒\n', das_time);

% --- 5. 融合通道并返回 ---
rc_data = sum(row_Th_column_Rh_data, 2);    
volume_RC = reshape(rc_data, [scan.N_z,scan.N_x,scan.N_y]);    

disp(' [Function] RC DAS 重建完成。');
end