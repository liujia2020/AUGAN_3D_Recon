% % ==========================================================
% % Master_Recon_Loop_DAS.m
% % (用于 DAS 重建)
% % ==========================================================
clc;
clear;
close all;

% --- 1. 定义统一的文件夹结构 ---
MasterOutputDir = 'Ultrasound_Simulation_Data_DAS'; % (使用新文件夹)
inputDir = 'Ultrasound_Simulation_Data\02_RF_Data'; % (数据源还是老地方)
outputDir_mat = fullfile(MasterOutputDir, '03_Recon_MAT');
outputDir_nii_orig = fullfile(MasterOutputDir, '04_NII_Original_Anisotropic'); 
outputDir_nii_resampled = fullfile(MasterOutputDir, '04_NII_Resampled_Isotropic'); 
outputDir_img = fullfile(MasterOutputDir, '05_Recon_Images');

% 自动创建所有输出文件夹
if ~exist(outputDir_mat, 'dir'), mkdir(outputDir_mat); end
if ~exist(outputDir_nii_orig, 'dir'), mkdir(outputDir_nii_orig); end 
if ~exist(outputDir_nii_resampled, 'dir'), mkdir(outputDir_nii_resampled); end 
if ~exist(outputDir_img, 'dir'), mkdir(outputDir_img); end

% --- 2. 查找并正确排序所有 RF 文件 ---
fileList = dir(fullfile(inputDir, 'SimData_RF_*.mat'));
if isempty(fileList)
    error('在 %s 中未找到 SimData_RF_*.mat 文件。', inputDir);
end
fileNames = {fileList.name};
[~, sortIdx] = sort(fileNames);
fileList = fileList(sortIdx);
fprintf('--- 自动化 DAS 重建开始，共找到 %d 个文件 ---\n', length(fileList));

% --- 3. (!!!) 定义【统一的】重建网格 (scan) ---
% (我们使用 1024x128x128 网格，与你的 CR 脚本和 FK 脚本一致)
pitch = 0.2e-3;
scan.N_x = 128;
scan.N_y = 128;
scan.x_axis = linspace(-127*pitch/2,127*pitch/2,scan.N_x);
scan.y_axis = linspace(-127*pitch/2,127*pitch/2,scan.N_y);
scan.startdepth = 5e-3;
scan.enddepth = 42e-3;
scan.N_z = 1024;
scan.z_axis = linspace(scan.startdepth,scan.enddepth,scan.N_z);
[scan.x,scan.z,scan.y] = meshgrid(scan.x_axis,scan.z_axis,scan.y_axis);
scan.N_pixels = size(scan.x(:),1);
disp('已定义统一重建网格 (1024x128x128)。');

% --- 4. 循环处理每个文件 ---
for i = 1:length(fileList)
    currentFile = fileList(i).name;
    fullFilePath = fullfile(inputDir, currentFile);
    fprintf('\n========================================\n');
    fprintf('正在检查: %s (%d / %d)\n', currentFile, i, length(fileList));
    
    % --- 5. 检查是否已重建 (基于文件名编号) ---
    [~, baseName, ~] = fileparts(currentFile);
    num_str = regexp(baseName, '\d+', 'match', 'once');
    if isempty(num_str)
        fprintf(' > 警告: 无法从 %s 中解析编号。跳过此文件。\n', currentFile);
        continue;
    end
    outputBaseName = sprintf('Recon_Combined_DAS_%s', num_str);
    expected_nii_file = fullfile(outputDir_nii_resampled, [outputBaseName, '.nii']);
    
    if exist(expected_nii_file, 'file')
        fprintf(' > 已跳过 (输出文件已存在): %s\n', outputBaseName);
        continue;
    end
    
    % --- 6. 加载 RF 数据 ---
    fprintf(' > 正在加载 RF 文件...\n');
    try
        loadedData = load(fullFilePath);
        disp(' > RF 文件加载成功。');
    catch ME
        warning(ME.identifier, '加载文件 %s 失败: %s. 跳过此文件。', currentFile, ME.message);
        continue;
    end
    
    % --- 7. 调用 DAS 重建函数 ---
    fprintf(' > 正在重建 (DAS)...\n');
    try
        % (!! 核心调用 !!)
        % (注意: 我们将定义好的 scan 结构体传入)
        [volume_CR] = function_Recon_CR_DAS(...
            loadedData.RcvData, loadedData.Trans, loadedData.Resource, ...
            loadedData.TX, loadedData.TW, loadedData.Receive, scan);
        
        % (!! 核心调用 !!)
        [volume_RC] = function_Recon_RC_DAS(...
            loadedData.RcvData, loadedData.Trans, loadedData.Resource, ...
            loadedData.TX, loadedData.TW, loadedData.Receive, scan);
            
    catch ME_recon
        warning(ME_recon.identifier, 'DAS 重建 %s 时出错: %s. 跳过此文件。', currentFile, ME_recon.message);
        disp('错误详情:');
        disp(ME_recon.getReport); 
        continue;
    end
    
    % --- 8. 融合 (正交复合) ---
    fprintf(' > 正在复合 RC 和 CR 图像 (DAS)...\n');
    volume_combined = volume_CR .* volume_RC;
    
    % (保存原始坐标轴的引用)
    x_space_original = scan.x_axis;
    y_space_original = scan.y_axis;
    z_space_original = scan.z_axis;
    
    % --- 8.5. 保存【原始】NIfTI 文件 (重采样之前) ---
    nii_filename_orig = fullfile(outputDir_nii_orig, [outputBaseName, '.nii']);
    fprintf(' > 正在保存【原始】NIfTI 文件 (DAS)...\n');
    try
        voxel_spacing_orig = [
            (z_space_original(2) - z_space_original(1)), ... 
            (x_space_original(2) - x_space_original(1)), ... 
            (y_space_original(2) - y_space_original(1))  ... 
        ] * 1000;
        
        volume_data_nii_orig = process_nii_image_local(volume_combined);
        
        niftiwrite(volume_data_nii_orig, nii_filename_orig, 'Compressed', false);
        info = niftiinfo(nii_filename_orig);
        info.PixelDimensions = voxel_spacing_orig; 
        info.SpaceUnits = 'Millimeter';
        info.Datatype = 'single';
        niftiwrite(volume_data_nii_orig, nii_filename_orig, info, 'Compressed', false);
        fprintf(' > 原始 .nii 文件已保存: %s\n', nii_filename_orig);
    catch ME_save_nii_orig
        warning(ME_save_nii_orig.identifier, '保存【原始】 .nii 文件失败: %s', ME_save_nii_orig.message);
    end

    % --- 9. 重采样 (使体素各向同性) ---
    % (DAS 网格也是各向异性的, z_voxel ~ 0.036mm, x/y_voxel ~ 0.2mm)
    Run_Resampling = true; 
    if Run_Resampling
        fprintf(' > 正在将 Z 轴从 %d 缩小以适配神经网络...\n', size(volume_combined, 1));
        try
            original_z_spacing_mm = (z_space_original(2) - z_space_original(1)) * 1000;
            target_spacing_mm = (x_space_original(2) - x_space_original(1)) * 1000;
            original_z_dim = size(volume_combined, 1);
            scale_factor = original_z_spacing_mm / target_spacing_mm;
            target_z_dim = round(original_z_dim * scale_factor);
            target_x_dim = size(volume_combined, 2);
            target_y_dim = size(volume_combined, 3);
            
            new_dims = [target_z_dim, target_x_dim, target_y_dim];
            fprintf(' > 目标维度: [%d, %d, %d]\n', new_dims(1), new_dims(2), new_dims(3));
            
            real_part_resampled = imresize3(real(volume_combined), new_dims, 'linear');
            imag_part_resampled = imresize3(imag(volume_combined), new_dims, 'linear');
            volume_combined = complex(real_part_resampled, imag_part_resampled);
            
            z_space_cropped = linspace(z_space_original(1), z_space_original(end), target_z_dim);
            x_space_cropped = x_space_original; 
            y_space_cropped = y_space_original;
            
            fprintf(' > 重采样完毕。新容积大小: %s\n', mat2str(size(volume_combined)));
        catch ME_resize
            warning(ME_resize.identifier, '重采样失败: %s. 将使用原始数据保存。', ME_resize.message);
            z_space_cropped = z_space_original;
            x_space_cropped = x_space_original;
            y_space_cropped = y_space_original;
        end
    end
    
    % --- 10. 保存最终结果 (重采样后) ---
    fprintf(' > 正在保存【重采样后】的 NIfTI, .mat, 和 .png 文件 (DAS)...\n');
    
    % (保存 .mat)
    mat_filename = fullfile(outputDir_mat, [outputBaseName, '.mat']);
    try
        save(mat_filename, ...
            'volume_combined', ...
            'x_space_cropped', 'y_space_cropped', 'z_space_cropped', ...
            '-v7.3');
        fprintf(' > 融合后的 .mat 文件已保存: %s\n', mat_filename);
    catch ME_save_mat
        warning(ME_save_mat.identifier, '保存 .mat 文件失败: %s', ME_save_mat.message);
    end
    
    % (保存 .nii 重采样)
    nii_filename_resampled = fullfile(outputDir_nii_resampled, [outputBaseName, '.nii']);
    try
        voxel_spacing = [
            (z_space_cropped(2) - z_space_cropped(1)), ...
            (x_space_cropped(2) - x_space_cropped(1)), ...
            (y_space_cropped(2) - y_space_cropped(1))
        ] * 1000;
        
        volume_data_nii = process_nii_image_local(volume_combined);
        
        niftiwrite(volume_data_nii, nii_filename_resampled, 'Compressed', false);
        info = niftiinfo(nii_filename_resampled); 
        info.PixelDimensions = voxel_spacing;
        info.SpaceUnits = 'Millimeter';
        info.Datatype = 'single';
        niftiwrite(volume_data_nii, nii_filename_resampled, info, 'Compressed', false); 
        fprintf(' > 融合后的 .nii 文件已保存: %s\n', nii_filename_resampled); 
    catch ME_save_nii
        warning(ME_save_nii.identifier, '保存 .nii 文件失败: %s', ME_save_nii.message);
    end
    
    % (保存 .png 预览图)
    img_filename = fullfile(outputDir_img, [outputBaseName, '.png']);
    try
        hFig = figure('Visible', 'off');
        set(hFig, 'Position', [100 100 1200 400]);
        
        subplot(1,3,1);
        [~, z_index] = min(abs(z_space_cropped - 0.020));
        img_xy = process_image_local(squeeze(volume_combined(z_index,:,:)));
        imagesc(x_space_cropped*1000, y_space_cropped*1000, img_xy.');
        xlabel('Lateral Dist[mm]'); ylabel('Elevation Dist[mm]');
        colormap gray; colorbar; axis image; caxis([-60 0]);
        
        subplot(1,3,2);
        [~, x_index] = min(abs(0 - x_space_cropped));
        img_yz = process_image_local(squeeze(volume_combined(:,x_index,:)));
        imagesc(y_space_cropped*1000, z_space_cropped*1000, img_yz);
        xlabel('Lateral Dist[mm]'); ylabel('Axial Dist[mm]');
        colormap gray; colorbar;
        xlim([-12.7 12.7]); ylim([z_space_cropped(1)*1000 z_space_cropped(end)*1000]);
        
        subplot(1,3,3);
        [~, y_index] = min(abs(0 - y_space_cropped));
        img_xz = process_image_local(squeeze(volume_combined(:,:,y_index)));
        imagesc(x_space_cropped*1000, z_space_cropped*1000, img_xz);
        xlabel('Elevation Dist[mm]'); ylabel('Axial Dist[mm]');
        colormap gray; colorbar;
        xlim([-12.7 12.7]); ylim([z_space_cropped(1)*1000 z_space_cropped(end)*1000]);
        
        set(findall(hFig,'type','colorbar'),'Limits',[-60 0]);
        print(hFig, img_filename, '-dpng', '-r300');
        close(hFig);
        fprintf(' > 预览图像已保存: %s\n', img_filename);
    catch ME_save_img
        warning(ME_save_img.identifier, '保存 .png 预览图失败: %s', ME_save_img.message);
        if exist('hFig', 'var') && ishandle(hFig)
            close(hFig);
        end
    end
    
    fprintf('文件 %s 处理完毕。\n', currentFile);
    
    % --- 11. 清理内存 ---
    clear loadedData volume_CR volume_RC volume_combined volume_data_nii volume_data_nii_orig;
    
end

fprintf('\n========================================\n');
fprintf('--- 自动化 DAS 重建全部完成！ ---\n');


% % ==========================================================
% % 局部函数 (Local Functions)
% % ==========================================================
function img_envelope = process_image_local(img)
    % 用于 PNG 预览图
    img_envelope = abs(img);
    max_val = max(img_envelope(:));
    if max_val == 0, max_val = 1; end
    img_envelope = 20*log10(img_envelope./max_val);
    img_envelope(img_envelope < -60) = -60;
end

function volume_data_nii = process_nii_image_local(volume_complex)
    % 用于 NIfTI 保存
    volume_data_nii = abs(volume_complex);
    max_val = max(volume_data_nii(:));
    if max_val == 0
        max_val = 1; % 避免 log10(0)
    end
    volume_data_nii = 20*log10(volume_data_nii./max_val);
    volume_data_nii(volume_data_nii < -60) = -60;
    volume_data_nii = single(volume_data_nii);
end