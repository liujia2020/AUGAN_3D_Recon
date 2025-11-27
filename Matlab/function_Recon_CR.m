% % ==========================================================
% % function_Recon_CR.m (V5 - 支持多角度集合)
% % ==========================================================
function [volume_CR, x_axis, y_axis, z_axis] = function_Recon_CR(RcvData, Trans, Resource, TX, TW, Receive, angle_set)
% % (!! 新增 !!) 接收第7个参数 'angle_set' ('full' or 'zero')

disp(' [Function] 正在执行: 列发行收 (CR) 重建...');

% --- 1. 从输入参数中提取基本参数 ---
f0 = double(Trans.frequency*1e6);    
fs = f0*Receive(1).samplesPerWave;   
c0 = Resource.Parameters.speedOfSound;
lambda = c0/f0;                      
ElementPos = Trans.ElementPos.*lambda;
initial_time = TW.peak/Trans.frequency/1e6;
% 设定物理空间扫描范围
pitch = 0.2e-3;   
scan.startdepth = 5e-3;
scan.enddepth = 42e-3;

% --- 2. 提取发射角度和 RF 数据 (CR 模式) ---
steer = zeros(length(TX),2);
for n_wave = 1:length(TX)
    steer(n_wave,:) = TX(n_wave).Steer;
end
column_waves = length(TX)/2;
beta = steer(column_waves+1:end,2);  % 列发射角度
% 读取原始数据
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
% [!! 关键区别 !!] 提取 列发(column_waves+1:end) 行收(1:128) 的数据
RF_data = permute(channel_RF(:,1:128,column_waves+1:end,1), [1 3 2]); % [1920x33x128]
RF_data = hilbert(RF_data);
disp(' [Function] RF 数据提取完毕。');

% --- 3. 设置 FFT 和 k 空间参数 ---
Nx = 128;     
dx = 0.0002;  
Nz = 1024;
Nt = Receive(1).endSample; % 1920
z_axis_temp = linspace(scan.startdepth,scan.enddepth,Nz).';
dz = z_axis_temp(2)-z_axis_temp(1);
cc = 1540;    
fHigh = 30e6;
fLow = 2;
nFFTt = 8*2^nextpow2(Nt);   
nFFTx = 2*2^nextpow2(Nx);   
nFFTz = 2*2^nextpow2(Nz);   % 2048
% 设置频率和波数轴
dOmega = fs/nFFTt;
omega = ((0:(nFFTt-1)) - floor(nFFTt/2))'*dOmega;
omega = ifftshift(omega);
dkx = 1/dx/nFFTx;
kx = ((0:(nFFTx-1)) - floor(nFFTx/2))*dkx;
kx = ifftshift(kx);
dkz = 1/dz; 
kz = (fLow)/(cc) + (0:(nFFTz-1))*(dkz/nFFTz); 
% 频带和消逝波处理
omegaBandIndex = (omega <= (fHigh)) & (omega >= (fLow)); 
omegaBand = omega(omegaBandIndex); 
[kx_matrix1,omegaBand_matrix] = meshgrid(kx,omegaBand);
isevanescent = abs(omegaBand_matrix)./abs(kx_matrix1)<cc;
disp(' [Function] FK 域初始化完成。');

% --- 4. FK 域处理 (核心循环) ---
fk_data = zeros(nFFTz,nFFTx,128); 
PkzkxX = zeros(nFFTz,nFFTx,128); % 预分配

% (!! 新增 !!) 根据 'angle_set' 参数选择循环索引
if strcmpi(angle_set, 'full')
    angle_indices = 1:column_waves; % 1:33
    disp(' [Function] 开始 CR 重建循环 (Full 33 Angles)...');
elseif strcmpi(angle_set, 'zero')
    angle_indices = 17; % 0 度角 (第50次采集 = 33 + 17)
    disp(' [Function] 开始 CR 重建循环 (Zero Angle Only)...');
else
    error("未知的 angle_set: '%s'. 必须是 'full' 或 'zero'。", angle_set);
end
tic;

% (!! 修改 !!) 循环
for ixt = angle_indices
    RFshow = RF_data(1:Nt,ixt,1:128); 
    RFo_y = fft(RFshow,nFFTt,1);     
    RFo_y = RFo_y(omegaBandIndex,:); 
    RFo_ky = fft(RFo_y,nFFTx,2);     
    
    % [!! 关键区别 !!] 角度补偿 (使用 Y 轴偏转角)
    angle = TX(ixt+column_waves).Steer(2);
    sinA = sin(angle);
    cosA = cos(angle);
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
        Akzkx = kz_matrix./K_all;
        Akzkx(isnan(Akzkx)) = 1;
        Pkzkx = Pkzkx.*Akzkx;
        Pkzkx(isnan(Pkzkx) | (OMEGA_st < omegaBand(1)) | (OMEGA_st > omegaBand(end))) = 0;
        
        % (CR 掩码逻辑 - 始终是正确的)
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
recon_time = toc;
fprintf(' [Function] CR 循环 (set: %s) 完成, 耗时: %.2f 秒\n', angle_set, recon_time);

% --- 5. 逆变换、裁切并准备输出 ---
fk_data2 = ifft(ifft(fk_data,[],1),[],2);
start_idx = 140;
x_half = 128;
fk_data_CR = fk_data2(start_idx:start_idx+1023, 1:x_half, :);

% --- 6. 更新物理空间范围并返回 ---
z_space_cropped = linspace(5e-3, 42e-3, size(fk_data_CR,1)); 
x_space_cropped = linspace(-127*pitch/2, 127*pitch/2, x_half);    
y_space_cropped = linspace(-127*pitch/2, 127*pitch/2, size(fk_data_CR,3));

disp(' [Function] CR 重建完成。正在返回数据...');
volume_CR = fk_data_CR;
x_axis = x_space_cropped;
y_axis = y_space_cropped;
z_axis = z_space_cropped;

end