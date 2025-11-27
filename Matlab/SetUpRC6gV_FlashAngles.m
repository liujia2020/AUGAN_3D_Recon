% Notice:
%   This file is provided by Verasonics to end users as a programming
%   example for the Verasonics Vantage Research Ultrasound System.
%   Verasonics makes no claims as to the functionality or intended
%   application of this program and the user assumes all responsibility
%   for its use
%
% File name: SetUpRC6gV_FlashAngles.m - Example of plane wave imaging with
%                                       steered transmits.
% Description:
%   Sequence programming file for 128x128 element Row/Column array, using plane
%   wave transmits with multiple steering angles.  Plane wave transmits are first
%   on columns with receives on rows, then transmits are along rows with receives on
%   columns. The script allow the user to choose whether to reconstruct 3
%   orthoganl planes or the whole volume using the ReconRegion variable
%   (1 = whole volume, 5 = three ortogonal planes)
%   Processing is asynchronous with respect to acquisition.
%
% Last update:
%   09/02/21: VTS-2538 updates to call computeTrans; formal name RC6gV

clear all
P.startDepth = 0;   % Acquisition depth in wavelengths
P.endDepth = 192;   % EndDepth in wavelengths

ReconRegion=5;
xyplane=100; % XY plane for reconstruction (in waveleghts)

na = 33;      % Set na = number of angles.
if (na > 1), dtheta = (30*pi/180)/(na-1); P.startAngle = -30*pi/180/2;
else dtheta = 0; P.startAngle=0; end % set dtheta to range over +/- 18 degrees.

% Define system parameters.
Resource.Parameters.numTransmit = 256;      % number of transmit channels.
Resource.Parameters.numRcvChannels = 256;    % number of receive channels.
Resource.Parameters.speedOfSound = 1540;    % set speed of sound in m/sec before calling computeTrans
Resource.Parameters.verbose = 2;
Resource.Parameters.initializeOnly = 0;
Resource.Parameters.simulateMode = 0;


%  Resource.Parameters.simulateMode = 1; %forces simulate mode, even if hardware is present.
%  Resource.Parameters.simulateMode = 2; %stops sequence and processes RcvData continuously.
% Resource.Parameters.waitForProcessing=1; %Set waitForProcessing for synchronous script

% Specify Trans structure array.
Trans.name = 'RC6gV'; % Vermon 6 MHz Row Column Array
Trans.units = 'wavelengths';
Trans = computeTrans(Trans);

% Specify PData structure array.
PData.PDelta = [Trans.spacing, Trans.spacing, 0.5];
PData.Size(1) = 128;
PData.Size(2) = 128;
PData.Size(3) = ceil((P.endDepth-P.startDepth)/PData.PDelta(3)); % startDepth, endDepth and pdelta set PData.Size(3).
PData.Origin = [-Trans.spacing*63.5,Trans.spacing*63.5,P.startDepth]; % x,y,z of upper lft crnr of page 1.
PData.Region(1) = struct('Shape',struct('Name','PData'));
PData(1).Region(2) = struct('Shape',struct('Name','Slice','Orientation','yz','oPAIntersect',PData.Origin(1)+PData.PDelta(2)*(PData.Size(2)-1)/2,'andWithPrev',1));
PData(1).Region(3) = struct('Shape',struct('Name','Slice','Orientation','xz','oPAIntersect',PData.Origin(2)-PData.PDelta(1)*(PData.Size(1)-1)/2,'andWithPrev',1));
PData(1).Region(4) = struct('Shape',struct('Name','Slice','Orientation','xy','oPAIntersect',xyplane,'andWithPrev',1));

PData.Region = computeRegions(PData);

PData.Region(5).PixelsLA = unique([PData.Region(2).PixelsLA; PData.Region(3).PixelsLA; PData.Region(4).PixelsLA]);
PData.Region(5).Shape.Name = 'Custom';
PData.Region(5).numPixels = length(PData.Region(5).PixelsLA);


% [!!] 关键修正：在 100 和 500 之间选择一个随机的点数
minPoints = 100;
maxPoints = 500;
numPoints = randi([minPoints, maxPoints]); % MATLAB 函数：在 [min, max] 间生成随机整数
fprintf('--- 本次仿真将使用 %d 个随机点靶 ---\n', numPoints);

% 1. 从 PData 和 Trans 结构中获取成像区域边界
x_bound = Trans.spacing * 63.5; % X 轴边界 (wavelengths)
y_bound = Trans.spacing * 63.5; % Y 轴边界 (wavelengths)
z_min = P.startDepth;           % Z 轴最小深度 (wavelengths)
z_max = P.endDepth;             % Z 轴最大深度 (wavelengths)

% 2. 生成随机坐标
x_coords = (rand(numPoints, 1) * 2 - 1) * x_bound;
y_coords = (rand(numPoints, 1) * 2 - 1) * y_bound;
z_coords = rand(numPoints, 1) * (z_max - z_min) + z_min;

% 3. 定义散射体幅度
amplitudes = ones(numPoints, 1);

% 4. 组装 Media.MP 矩阵
Media.MP = [x_coords, y_coords, z_coords, amplitudes];

% 5. 设置 Media.numPoints
Media.numPoints = numPoints;

% 6. 保留原始脚本中的最后两行 [cite: 139, 140]
Media.attenuation = -0.5;
Media.function = 'movePoints';



% Specify Resources.
Resource.RcvBuffer(1).datatype = 'int16';
Resource.RcvBuffer(1).rowsPerFrame = na*4096; % this size allows for maximum range
Resource.RcvBuffer(1).colsPerFrame = Resource.Parameters.numRcvChannels;
Resource.RcvBuffer(1).numFrames = 10;    % 10 frames stored in RcvBuffer.
Resource.InterBuffer(1).numFrames = 1;   % one intermediate buffer needed.
Resource.ImageBuffer(1).numFrames = 1;


Resource.DisplayWindow(1).Type = 'Matlab';
Resource.DisplayWindow(1).Title = '3D RC6gV FlashAngles - XZ plane';
Resource.DisplayWindow(1).pdelta = 0.4;
Resource.DisplayWindow(1).Position = [0,40, ...
    ceil(PData(1).Size(2)*PData(1).PDelta(2)/Resource.DisplayWindow(1).pdelta), ... % width
    ceil(PData(1).Size(3)*PData(1).PDelta(3)/Resource.DisplayWindow(1).pdelta)];    % height
Resource.DisplayWindow(1).Orientation = 'xz';
Resource.DisplayWindow(1).ReferencePt = [PData(1).Origin(1),0,0.0];
Resource.DisplayWindow(1).Colormap = gray(256);
Resource.DisplayWindow(1).AxesUnits = 'wavelengths';
Resource.DisplayWindow(1).numFrames = 20;

Resource.DisplayWindow(2).Type = 'Matlab';
Resource.DisplayWindow(2).Title = '3D RC6gV FlashAngles - YZ plane';
Resource.DisplayWindow(2).pdelta = Resource.DisplayWindow(1).pdelta;
Resource.DisplayWindow(2).Position = [430,40, ...
    ceil(PData(1).Size(1)*PData(1).PDelta(1)/Resource.DisplayWindow(2).pdelta), ... % width
    ceil(PData(1).Size(3)*PData(1).PDelta(3)/Resource.DisplayWindow(2).pdelta)];    % height
Resource.DisplayWindow(2).Orientation = 'yz';
Resource.DisplayWindow(2).ReferencePt = [0,-PData(1).Origin(2),0];
Resource.DisplayWindow(2).Colormap = gray(256);
Resource.DisplayWindow(2).AxesUnits = 'wavelengths';
Resource.DisplayWindow(2).numFrames = 20;

Resource.DisplayWindow(3).Type = 'Matlab';
Resource.DisplayWindow(3).Title = '3D RC6gV FlashAngles - XY plane';
Resource.DisplayWindow(3).pdelta = Resource.DisplayWindow(1).pdelta;
Resource.DisplayWindow(3).Position = [0,40, ...
    ceil(PData(1).Size(2)*PData(1).PDelta(2)/Resource.DisplayWindow(3).pdelta), ... % width
    ceil(PData(1).Size(1)*PData(1).PDelta(1)/Resource.DisplayWindow(3).pdelta)];    % height
Resource.DisplayWindow(3).Orientation = 'xy';
Resource.DisplayWindow(3).ReferencePt = [PData(1).Origin(1),-PData(1).Origin(2),xyplane];%PData.Region(end).Shape.oPAIntersect];
Resource.DisplayWindow(3).Colormap = gray(256);
Resource.DisplayWindow(3).AxesUnits = 'wavelengths';
Resource.DisplayWindow(3).numFrames = 20;

% Specify TW structure array.
TW.type = 'parametric';
TW.Parameters = [Trans.frequency,0.67,2,1];

% Specify TX structure array. Define steered plane wave transmits for the elements
% along the x and y axes.
TX = repmat(struct('waveform', 1, ...
                   'Origin', [0.0,0.0,0.0], ...
                   'Apod', zeros(1,256), ...
                   'focus', 0.0, ...
                   'Steer', [0.0,0.0], ...
                   'Delay', zeros(1,Trans.numelements)), 1, 2*na);
% - Set event specific TX attributes.
if fix(na/2) == na/2       % if na even
    P.startAngle = (-(fix(na/2) - 1) - 0.5)*dtheta;
else
    P.startAngle = -fix(na/2)*dtheta;
end
for n = 1:na   % for na transmit events on x elements
    TX(n).Apod(1:128) = 1;
    TX(n).Steer = [(P.startAngle+(n-1)*dtheta),0.0];
    TX(n).Delay = computeTXDelays(TX(n));
end
k = na;
for n = 1:na   % for na transmits on y elements
    TX(k+n).Apod(129:256) = 1;
    TX(k+n).Steer = [0.0,(P.startAngle+(n-1)*dtheta)];
    TX(k+n).Delay = computeTXDelays(TX(k+n));
end

% Specify TGC Waveform structure.
TGC.CntrlPts = [0,245,424,515,627,661,717,744];
TGC.CntrlPts = [0 785.2216 1023 1023 1023 1023 1023 1023];
TGC.rangeMax = P.endDepth;
TGC.Waveform = computeTGCWaveform(TGC);

% Specify Receive structure arrays.
% - We need na Receives for each of the 2*na transmits.
maxAcqLength = ceil(sqrt(P.endDepth^2 + (128*Trans.spacing)^2));
Receive = repmat(struct('Apod', zeros(1,256), ...
                        'startDepth', P.startDepth, ...
                        'endDepth', maxAcqLength,...
                        'TGC', 1, ...
                        'bufnum', 1, ...
                        'framenum', 1, ...
                        'acqNum', 1, ...
                        'sampleMode', 'NS200BW', ...
                        'mode', 0, ...
                        'callMediaFunc', 0), 1, 2*na*Resource.RcvBuffer(1).numFrames);

% - Set event specific Receive attributes for each frame.
for i = 1:Resource.RcvBuffer(1).numFrames
    %     Receive(na*(i-1)+1).callMediaFunc = 1;
    m = 2*na*(i-1); % m counts frames
    for j = 1:na % for transmits on x elements, receive on y
        Receive(m+j).Apod(129:256) = 1; 
        Receive(m+j).framenum = i;
        Receive(m+j).acqNum = j;
    end
    for j = 1:na % for transmits on y elements, receive on x
        Receive(m+na+j).Apod(1:128) = 1;
        Receive(m+na+j).framenum = i;
        Receive(m+na+j).acqNum = na+j;
    end
end

% Specify Recon structure arrays.
% - We need one Recon structure which will reconstruct the entire volume.
Recon = repmat(struct('senscutoff', 0.6, ...
                      'pdatanum', 1, ...
                      'rcvBufFrame',-1, ...
                      'IntBufDest', [1,1], ...
                      'ImgBufDest', [1,-1], ...
                      'RINums', 1:(2*na)), 1, 1);
% Define ReconInfo structures.
% We need 2*na ReconInfo structures for na steering angles.
ReconInfo = repmat(struct('mode', 'accumIQ', ...  % default is to accumulate IQ data.
                   'txnum', 1, ...
                   'rcvnum', 1, ...
                   'scaleFactor', 0.5, ...
                   'regionnum', 1), 1, 2*na);
% - Set specific ReconInfo attributes.
ReconInfo(1).mode = 'replaceIQ'; % replace IQ data
for j = 1:na  % For each transmit along x elements
    ReconInfo(j).txnum = j;
    ReconInfo(j).rcvnum = j;
    ReconInfo(j).regionnum = ReconRegion; %1 for the whole volume, 5 for the slices
end
% ReconInfo(na).Post = 'IQ2IntensityImageBuf';
% ReconInfo(na+1).Pre = 'clearInterBuf';
for j = 1:na  % For each transmit along y elements
    ReconInfo(na+j).txnum = na+j;
    ReconInfo(na+j).rcvnum = na+j;
    ReconInfo(na+j).regionnum = ReconRegion; %1 for the whole volume, 5 for the slices
end
ReconInfo(2*na).mode = 'accumIQ_replaceIntensity';
% ReconInfo(2*na).Post = 'IQ2IntensityImageBufAdd' ;

% Specify Process structure array.
pers = 20;

Process(1).classname = 'Image';
Process(1).method = 'imageDisplay';
Process(1).Parameters = {'imgbufnum',1,...   % number of buffer to process.
                         'framenum',-1,...   % (-1 => lastFrame)
                         'pdatanum',1,...    % number of PData structure to use
                         'srcData','intensity3D',...
                         'pgain',4.5,...     % pgain is image processing gain
                         'reject',2,...      % reject level
                         'persistMethod','simple',...
                         'persistLevel',pers,...
                         'interpMethod','4pt',...
                         'grainRemoval','none',...
                         'processMethod','none',...
                         'averageMethod','none',...
                         'compressMethod','power',...
                         'compressFactor',40,...
                         'mappingMethod','full',...
                         'display',1,...      % display image after processing
                         'displayWindow',1};
Process(2).classname = 'Image';
Process(2).method = 'imageDisplay';
Process(2).Parameters = {'imgbufnum',1,...   % number of buffer to process.
                         'framenum',-1,...   % (-1 => lastFrame)
                         'pdatanum',1,...    % number of PData structure to use
                         'srcData','intensity3D',...
                         'pgain',4.5,...     % pgain is image processing gain
                         'reject',2,...      % reject level
                         'persistMethod','simple',...
                         'persistLevel',pers,...
                         'interpMethod','4pt',...
                         'grainRemoval','none',...
                         'processMethod','none',...
                         'averageMethod','none',...
                         'compressMethod','power',...
                         'compressFactor',40,...
                         'mappingMethod','full',...
                         'display',1,...      % display image after processing
                         'displayWindow',2};
Process(3).classname = 'Image';
Process(3).method = 'imageDisplay';
Process(3).Parameters = {'imgbufnum',1,...   % number of buffer to process.
                         'framenum',-1,...   % (-1 => lastFrame)
                         'pdatanum',1,...    % number of PData structure to use
                         'srcData','intensity3D',...
                         'pgain',4.5,...     % pgain is image processing gain
                         'reject',2,...      % reject level
                         'persistMethod','simple',...
                         'persistLevel',pers,...
                         'interpMethod','4pt',...
                         'grainRemoval','none',...
                         'processMethod','none',...
                         'averageMethod','none',...
                         'compressMethod','power',...
                         'compressFactor',40,...
                         'mappingMethod','full',...
                         'display',1,...      % display image after processing
                         'displayWindow',3};


% Specify SeqControl structure arrays.
SeqControl(1).command = 'jump'; % jump back to start
SeqControl(1).argument = 1;
SeqControl(2).command = 'timeToNextAcq';  % time between synthetic aperture acquisitions
SeqControl(2).argument = 150;  % 160 usec
SeqControl(3).command = 'timeToNextAcq';  % time between frames
SeqControl(3).argument = 20000 - (2*na-1)*150;  % 50 msec (50 Hz)
SeqControl(4).command = 'returnToMatlab';
nsc = 5; % nsc is count of SeqControl objects

% Specify Event structure arrays.
n = 1;
for i = 1:Resource.RcvBuffer(1).numFrames
    m = 2*na*(i-1);
    for j = 1:na   % Acquire x element transmits
        Event(n).info = 'x elements';
        Event(n).tx = j;
        Event(n).rcv = m+j;
        Event(n).recon = 0;
        Event(n).process = 0;
        Event(n).seqControl = 2;
        n = n+1;
        
        Event(n).info = 'y elements';
        Event(n).tx = na+j;
        Event(n).rcv = m+na+j;
        Event(n).recon = 0;
        Event(n).process = 0;
        Event(n).seqControl = 2;
        n = n+1;
    end
    Event(n-1).seqControl = [3,nsc]; % modify last acquisition Event's seqControl
    SeqControl(nsc).command = 'transferToHost'; % transfer frame to host buffer
    nsc = nsc+1;

    Event(n).info = 'recon ';
    Event(n).tx = 0;
    Event(n).rcv = 0;
    Event(n).recon = 1;
    Event(n).process = 0;
    Event(n).seqControl = 0;
    n = n+1;

      Event(n).info = ['Process XZ - Frame ' num2str(j)];
    Event(n).tx = 0;
    Event(n).rcv = 0;
    Event(n).recon = 0;
    Event(n).process = 1;
    Event(n).seqControl = 0;
    n = n+1;

    Event(n).info = ['Process YZ - Frame ' num2str(j)];
    Event(n).tx = 0;
    Event(n).rcv = 0;
    Event(n).recon = 0;
    Event(n).process = 2;
    Event(n).seqControl = 0;

    n = n+1;

    Event(n).info = ['Process XY - Frame ' num2str(j)];
    Event(n).tx = 0;
    Event(n).rcv = 0;
    Event(n).recon = 0;
    Event(n).process = 3;
    Event(n).seqControl = 0;
    if floor(i/3) == i/3     % Exit to Matlab every 3rd frame reconstructed
        Event(n).seqControl = 4;
    end
    n = n+1;
end

Event(n).info = 'Jump back';
Event(n).tx = 0;
Event(n).rcv = 0;
Event(n).recon = 0;
Event(n).process = 0;
Event(n).seqControl = 4;


%% User specified UI Control Elements
import vsv.seq.uicontrol.VsSliderControl;

% - Sensitivity Cutoff
UI(1).Control = VsSliderControl('LocationCode','UserB7','Label','Sens. Cutoff',...
                  'SliderMinMaxVal',[0,1.0,Recon(1).senscutoff],...
                  'SliderStep',[0.025,0.1],'ValueFormat','%1.3f', ...
                  'Callback', @SensCutoffCallback );

% Specify factor for converting sequenceRate to frameRate.
frameRateFactor = 3;

% ==========================================================
% == 核心修改 2: 保存为带编号和点数(Points)的唯一文件 ==
% ==========================================================
MasterOutputDir = 'Ultrasound_Simulation_Data_500_2'; % 定义主文件夹
outputDir = fullfile(MasterOutputDir, '01_Config_Files'); % 定义子文件夹
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('已创建新目录: %s\n', outputDir);
end
% --- 自动查找下一个文件编号 ---
prefix = 'Config_Scene_';
suffix = '.mat';
% file_list = dir(fullfile(outputDir, [prefix, '*.mat'])); % 查找所有匹配的文件
% max_num = 0;
% if ~isempty(file_list)
%     for k = 1:length(file_list)
%         filename = file_list(k).name;
%         % [!!] 使用正则表达式从 'Config_Scene_0001_Pts_342.mat' 中提取 '0001'
%         num_tokens = regexp(filename, [prefix, '(\d+)'], 'tokens');
%         if ~isempty(num_tokens)
%             num = str2double(num_tokens{1}{1});
%             if ~isnan(num) && num > max_num
%                 max_num = num;
%             end
%         end
%     end
% end
% --- (已修正) 自动查找下一个文件编号 ---
file_list = dir(fullfile(outputDir, [prefix, '*', suffix]));
max_num = 0;
if ~isempty(file_list)
    for k = 1:length(file_list)
        filename = file_list(k).name;
        % [!!] 使用正则表达式从 'Config_Scene_0001_Pts_342.mat' 中提取 '0001'
        num_tokens = regexp(filename, [prefix, '(\d+)'], 'tokens');
        if ~isempty(num_tokens)
            num = str2double(num_tokens{1}{1});
            if ~isnan(num) && num > max_num
                max_num = num;
            end
        end
    end
end
next_num = max_num + 1;
% ---------------------------------
% [!!] 核心修改：在文件名中加入 numPoints [!!]
if ~exist('numPoints', 'var')
    error('numPoints 变量未定义。请在保存前先生成 numPoints。');
end

outputFilename = sprintf('%s%04d_Pts_%03d%s', prefix, next_num, numPoints, suffix); 
% 示例: 'Config_Scene_0001_Pts_342.mat'

fullFilePath = fullfile(outputDir, outputFilename);
fprintf('正在保存配置文件到: %s\n', fullFilePath);
save(fullFilePath); % 保存所有变量到新文件

% --- (!! 新增 !!) 创建通信文件，告诉 Save_RF_Data 脚本我们用了哪个文件名 ---
% (我们把这个文件保存在主文件夹中)
comm_file = fullfile(MasterOutputDir, '__latest_config.txt');
try
    fid = fopen(comm_file, 'w');
    fprintf(fid, '%s', outputFilename); % 只保存文件名，不带路径
    fclose(fid);
    fprintf('已创建通信文件: %s\n', comm_file);
catch ME_comm
    error('创建通信文件 %s 失败: %s', comm_file, ME_comm.message);
end


fprintf('\n--- 配置文件已成功生成！---\n');
fprintf('文件名: %s\n', fullFilePath);

% ==========================================================
% == 核心解决方案：创建 'filename' 变量供 VSX 读取 ==
% ==========================================================
filename = fullFilePath;

% ==========================================================

% == 保存修改结束 ==

%% **** Callback routines to be converted by text2cell function. ****

function SensCutoffCallback(~,~,UIValue)
    %SensCutoff - Sensitivity cutoff change
    ReconL = evalin('base', 'Recon');
    for i = 1:size(ReconL,2)
        ReconL(i).senscutoff = UIValue;
    end
    assignin('base','Recon',ReconL);
    Control = evalin('base','Control');
    Control.Command = 'update&Run';
    Control.Parameters = {'Recon'};
    assignin('base','Control', Control);
end